"""Lab 3: Semantic uncertainty (semantic entropy, KLE, probes).

End-to-end demo:
  1) Sample S responses per prompt from an LLM pool (broadcast).
  2) Embed each response using an embedding pool (/v1/embeddings).
  3) Compute uncertainty per prompt using:
       - cluster-based semantic entropy, or
       - kernel language entropy (KLE), or
       - a pre-trained semantic entropy probe (optional).

USAGE (LLM + embedding):
  python examples/lab3_semantic_uncertainty.py \\
    --hostfile agents.txt \\
    --embed-hostfile embed_agents.txt \\
    --prompt "Summarize the main claim in one sentence."

If --embed-hostfile is omitted, the script uses --hostfile endpoints filtered
by --embed-role (default: role=embed). This mirrors examples/test_embeddings_sg.py.

NOTE on probes:
  The probe path expects hidden-state features to be supplied externally
  (e.g. from a model/API that returns hidden states). This example supports
  reading those features from a .npy file; it does not extract hidden states
  from vLLM by default.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from aurora_swarm import EmbeddingPool, VLLMPool, parse_hostfile
from aurora_swarm.patterns.embedding import scatter_gather_embeddings
from aurora_swarm.patterns.scatter_gather import scatter_gather
from aurora_swarm.uq import (
    kernel_language_entropy,
    load_probe,
    predict_semantic_entropy,
    semantic_entropy,
)


def print_ts(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr)


@dataclass
class PromptResult:
    prompt_id: int
    prompt: str
    responses: list[str]
    embeddings: list[list[float]]
    uncertainty: float


def _read_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt is not None:
        return [args.prompt]
    if args.prompts is not None:
        p = Path(args.prompts)
        lines = [ln.strip() for ln in p.read_text().splitlines()]
        return [ln for ln in lines if ln and not ln.startswith("#")]
    raise ValueError("Need --prompt or --prompts")


def _resolve_hostfile(path: Path | None) -> Path:
    if path is not None:
        return path
    env = os.environ.get("AURORA_SWARM_HOSTFILE")
    if not env:
        raise ValueError("Need --hostfile or AURORA_SWARM_HOSTFILE")
    return Path(env)


def _resolve_embed_endpoints(
    embed_hostfile: Path | None,
    llm_hostfile: Path,
    embed_role: str,
) -> list:
    """Return list of AgentEndpoint for embedding (from parse_hostfile + optional role filter)."""
    if embed_hostfile is None:
        endpoints = parse_hostfile(llm_hostfile)
        if embed_role:
            endpoints = [ep for ep in endpoints if ep.tags.get("role") == embed_role]
    else:
        endpoints = parse_hostfile(embed_hostfile)
    if not endpoints:
        raise ValueError("No embedding endpoints resolved (check --embed-hostfile or --embed-role).")
    return endpoints


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--hostfile", type=Path, help="LLM hostfile (default: AURORA_SWARM_HOSTFILE).")
    parser.add_argument("--embed-hostfile", type=Path, help="Embedding hostfile (optional).")
    parser.add_argument(
        "--embed-role",
        default="embed",
        help="If --embed-hostfile omitted, filter endpoints by role tag (default: embed). Use empty string to disable.",
    )
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="LLM model id for vLLM pool.")
    parser.add_argument(
        "--embed-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model id for /v1/embeddings.",
    )
    parser.add_argument(
        "--embed-max-tokens",
        type=int,
        default=128,
        help="Approx token cap for each text sent to embeddings (chars≈4*tokens) (default: 128).",
    )
    parser.add_argument("--prompt", type=str, help="Single prompt to evaluate.")
    parser.add_argument("--prompts", type=Path, help="File with one prompt per line (# comments allowed).")
    parser.add_argument("--num-samples", type=int, default=8, help="S: number of samples per prompt (default: 8).")
    parser.add_argument(
        "--method",
        choices=["semantic_entropy", "kle", "probe"],
        default="semantic_entropy",
        help="Uncertainty method (default: semantic_entropy).",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.9,
        help="Cosine similarity threshold for semantic entropy clustering (default: 0.9).",
    )
    parser.add_argument(
        "--kle-kernel",
        choices=["cosine", "rbf"],
        default="cosine",
        help="Kernel for KLE (default: cosine).",
    )
    parser.add_argument(
        "--probe",
        type=Path,
        help="Path to saved probe .npz (required for --method probe).",
    )
    parser.add_argument(
        "--hidden-states",
        type=Path,
        help="Path to .npy hidden-state feature matrix (n_prompts, n_features), required for --method probe.",
    )
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout seconds (default: 300).")
    parser.add_argument("--concurrency", type=int, default=64, help="Max concurrent requests (default: 64).")
    parser.add_argument("--max-tokens", type=int, default=512, help="LLM max_tokens per sample (default: 512).")
    parser.add_argument("--output", type=Path, help="Write results JSONL to this path.")
    args = parser.parse_args()

    llm_hostfile = _resolve_hostfile(args.hostfile).resolve()
    prompts = _read_prompts(args)
    S = max(1, int(args.num_samples))

    print_ts("=" * 60)
    print_ts("Lab 3: Semantic uncertainty")
    print_ts("=" * 60)
    # We'll print llm_endpoints after we resolve filtering below.
    print_ts(f"LLM hostfile: {llm_hostfile}  prompts: {len(prompts)}  S: {S}  method: {args.method}")

    embed_endpoints = _resolve_embed_endpoints(
        args.embed_hostfile.resolve() if args.embed_hostfile else None,
        llm_hostfile,
        args.embed_role,
    )
    embed_pool = EmbeddingPool(
        embed_endpoints,
        model=args.embed_model,
        timeout=args.timeout,
        concurrency=args.concurrency,
    )
    print_ts(f"Embedding endpoints: {len(embed_endpoints)}  embed_model: {args.embed_model}")

    # LLM pool. We generate S samples per prompt by sending the same prompt
    # S times round-robin across available endpoints (so S can exceed endpoint count).
    #
    # IMPORTANT: if we are using role-tagged embedding endpoints (role=embed by default)
    # from the same hostfile, exclude them from the LLM sampling pool so we don't send
    # generation prompts to embedding-only servers.
    endpoints_all = parse_hostfile(llm_hostfile)
    if not endpoints_all:
        print("Error: No LLM endpoints in hostfile.", file=sys.stderr)
        return 1
    if args.embed_hostfile is None and args.embed_role:
        endpoints = [ep for ep in endpoints_all if ep.tags.get("role") != args.embed_role]
    else:
        endpoints = endpoints_all
    if not endpoints:
        print(
            f"Error: After excluding embed_role={args.embed_role!r}, no LLM endpoints remain. "
            "Provide a separate --embed-hostfile or adjust --embed-role.",
            file=sys.stderr,
        )
        return 1
    print_ts(f"LLM endpoints (non-embed): {len(endpoints)}")
    pool = VLLMPool(
        endpoints,
        model=args.model,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        concurrency=args.concurrency,
    )

    results: list[PromptResult] = []
    sub_pool = None
    try:
        # Use all endpoints for sampling throughput; samples are distributed round-robin.
        sub_pool = pool

        # If probe method, load probe + hidden states up front.
        probe_coef = probe_intercept = probe_meta = None
        hidden_states = None
        if args.method == "probe":
            if not args.probe or not args.hidden_states:
                print("Error: --method probe requires --probe and --hidden-states.", file=sys.stderr)
                return 1
            probe_coef, probe_intercept, probe_meta = load_probe(args.probe)
            hidden_states = np.load(args.hidden_states)
            if hidden_states.shape[0] != len(prompts):
                print(
                    f"Error: hidden-states rows ({hidden_states.shape[0]}) must match number of prompts ({len(prompts)}).",
                    file=sys.stderr,
                )
                return 1
            print_ts(f"Loaded probe: {args.probe}  meta: {probe_meta or {}}")

        # Process each prompt: generate S samples, embed them, compute uncertainty.
        async with embed_pool:
            for pid, prompt in enumerate(prompts):
                print_ts(f"[{pid}] sampling {S} responses across {sub_pool.size} endpoint(s)...")
                responses = await scatter_gather(sub_pool, [prompt] * S)
                texts = [r.text for r in responses if r.success and (r.text or "").strip()]
                if not texts:
                    results.append(PromptResult(pid, prompt, [], [], float("nan")))
                    continue

                # Embedding endpoints often have strict max input lengths. Truncate each response
                # to avoid 400 errors; use a conservative chars≈4*tokens heuristic.
                max_chars = max(1, int(args.embed_max_tokens) * 4)
                texts_for_embed = [
                    (t[:max_chars].rsplit("\n", 1)[0] if ("\n" in t[:max_chars]) else t[:max_chars])
                    for t in texts
                ]
                responses_embed = await scatter_gather_embeddings(embed_pool, texts_for_embed)
                emb_ok = [r.embedding for r in responses_embed if r.success and r.embedding is not None]
                if len(emb_ok) == 0:
                    print_ts(f"[{pid}] embedding failures: {len(texts) - len(emb_ok)}/{len(texts)} (all failed)")
                    results.append(PromptResult(pid, prompt, texts, [], float("nan")))
                    continue

                E = np.asarray(emb_ok, dtype=np.float64)
                if args.method == "semantic_entropy":
                    uq = semantic_entropy(E, similarity_threshold=args.similarity_threshold)
                elif args.method == "kle":
                    uq = kernel_language_entropy(E, kernel=args.kle_kernel)
                else:
                    # probe
                    uq = float(predict_semantic_entropy(hidden_states[pid], probe_coef, float(probe_intercept))[0])

                results.append(PromptResult(pid, prompt, texts, emb_ok, uq))
                print_ts(f"[{pid}] uncertainty={uq:.4f}  responses={len(texts)}  emb_ok={len(emb_ok)}")
    finally:
        if sub_pool is not None:
            await sub_pool.close()
        await pool.close()
        await asyncio.sleep(0)

    # Emit results
    out_lines: list[str] = []
    for r in results:
        out_lines.append(
            json.dumps(
                {
                    "prompt_id": r.prompt_id,
                    "prompt": r.prompt,
                    "num_responses": len(r.responses),
                    "num_embeddings": len(r.embeddings),
                    "uncertainty": r.uncertainty,
                    "method": args.method,
                },
                ensure_ascii=False,
            )
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(out_lines) + "\n")
        print_ts(f"Wrote {len(out_lines)} results to {args.output}")
    else:
        print("\n".join(out_lines))

    print_ts("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

