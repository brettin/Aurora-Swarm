"""Scatter-Gather pattern example using TOM.COLI bacterial gene dataset.

This script demonstrates scaling LLM inference using Aurora-Swarm's scatter_gather
pattern across multiple vLLM endpoints. It processes bacterial gene data and queries
an LLM about gene properties, functions, and research opportunities.

COMPARISON TO test.coli_v3.py:
------------------------------
The Aurora-Inferencing/examples/TOM.COLI/test.coli_v3.py script uses:
  - OpenAI client with synchronous API calls
  - ThreadPoolExecutor for parallelism
  - Manual batch management

This script uses:
  - Aurora-Swarm's async scatter_gather pattern
  - VLLMPool for connection pooling and concurrency control
  - Automatic round-robin distribution across agents

BENEFITS:
---------
  - True async/await concurrency (no thread overhead)
  - Automatic load balancing across N endpoints
  - Built-in connection pooling and semaphore control
  - Cleaner code with pattern abstractions

INPUT DATA FORMAT:
------------------
Tab-separated values with format:
    genome_id<TAB>organism<TAB>gene_id<TAB>gene_description

Example:
    100177.28<TAB>Borreliella lusitaniae<TAB>QIA24_00005<TAB>Acylphosphatase

HOSTFILE FORMAT:
----------------
One endpoint per line (see aurora_swarm.hostfile documentation):
    hostname1:8000
    hostname2:8001
    hostname3:8000 node=worker-01

USAGE EXAMPLES:
---------------

Using a hostfile:
    python examples/scatter_gather_coli.py \\
        /path/to/batch_1/ \\
        --hostfile agents.txt \\
        --num-files 10 \\
        --output results.txt

Using environment variable:
    export AURORA_SWARM_HOSTFILE=/path/to/agents.txt
    python examples/scatter_gather_coli.py /path/to/batch_1/

Process first 100 chunk files:
    python examples/scatter_gather_coli.py \\
        /path/to/batch_1/ \\
        --num-files 100 \\
        --hostfile agents.txt \\
        --max-tokens 2048 \\
        --model meta-llama/Llama-3.1-70B-Instruct

PERFORMANCE:
------------
Scales linearly with number of endpoints. With 100 endpoints processing 10,000
prompts, expect throughput of ~1000-2000 prompts/minute depending on model size
and max_tokens configuration.
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

from aurora_swarm import VLLMPool, parse_hostfile
from aurora_swarm.patterns.scatter_gather import scatter_gather


def print_with_timestamp(message: str) -> None:
    """Print message with timestamp prefix."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr)


def discover_chunk_files(input_dir: Path, num_files: int, skip_files: int = 0) -> list[Path]:
    """Discover chunk files in input directory.
    
    Parameters
    ----------
    input_dir:
        Directory containing chunk_*.txt files.
    num_files:
        Maximum number of files to return.
    skip_files:
        Number of files to skip at the start (for resuming).
    
    Returns
    -------
    list[Path]
        List of chunk file paths, sorted by name.
    """
    chunk_files = sorted(input_dir.glob("chunk_*.txt"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.txt files found in {input_dir}")
    
    # Skip files and take the requested number
    start_idx = skip_files
    end_idx = skip_files + num_files
    return chunk_files[start_idx:end_idx]


def parse_gene_line(line: str, line_num: int) -> tuple[str, str, str, str] | None:
    """Parse a TSV line containing gene information.
    
    Parameters
    ----------
    line:
        Tab-separated line with genome_id, organism, gene_id, gene_description.
    line_num:
        Line number for error reporting.
    
    Returns
    -------
    tuple or None
        (genome_id, organism, gene_id, gene_description) or None if parse fails.
    """
    line = line.strip()
    if not line:
        return None
    
    parts = line.split('\t')
    if len(parts) < 4:
        print_with_timestamp(
            f"Warning: Line {line_num} has fewer than 4 fields, skipping: {line[:50]}..."
        )
        return None
    
    genome_id = parts[0]
    organism = parts[1]
    gene_id = parts[2]
    gene_description = '\t'.join(parts[3:])  # In case description contains tabs
    
    return genome_id, organism, gene_id, gene_description


def construct_prompt(organism: str, gene_id: str, gene_description: str) -> str:
    """Construct LLM prompt for bacterial gene analysis.
    
    Uses the same prompt template as Aurora-Inferencing test.coli_v3.py.
    
    Parameters
    ----------
    organism:
        Organism name (e.g., "Borreliella lusitaniae").
    gene_id:
        Gene identifier.
    gene_description:
        Gene description or function.
    
    Returns
    -------
    str
        Formatted prompt for the LLM.
    """
    gene_data = f"{gene_id}\t{gene_description}"
    
    prompt = (
        "Please tell me (using the knowledge you have been trained on) what you know about this bacterial gene in "
        + organism
        + " whose various IDs are given here, though they all refer to the same gene: "
        + gene_data
        + ". In particular, we want to know the following information: Is this gene well studied or is it hypothetical with unknown function? "
        "Is the gene essential for survival? Is the gene or gene product a good antibacterial drug target? What other genes does this gene interact with? "
        "Is this gene part of an operon (cluster of genes on the chromosome that work together to carry out complex functions)? "
        "Is this gene involved in transcriptional regulation? Is it known what gene regulates this gene's expression? "
        "Does this gene also occur in other bacteria? If you were starting out as a research microbiologist, what might be a hypothesis you could explore related to this protein that would have significant scientific impact? "
        "Where possible, give concise answers to these questions as well as describe the function of the gene more generally if it is known."
    )
    
    return prompt


def read_and_prepare_data(chunk_files: list[Path]) -> tuple[list[str], list[str], list[str], list[str]]:
    """Read chunk files and prepare prompts and metadata.
    
    Parameters
    ----------
    chunk_files:
        List of paths to chunk files.
    
    Returns
    -------
    tuple
        (prompts, genome_ids, gene_ids, organisms) lists with matching indices.
    """
    prompts = []
    genome_ids = []
    gene_ids = []
    organisms = []
    
    total_lines = 0
    skipped_lines = 0
    
    print_with_timestamp(f"Reading {len(chunk_files)} chunk files...")
    
    for file_idx, file_path in enumerate(chunk_files, 1):
        print_with_timestamp(f"  [{file_idx}/{len(chunk_files)}] Processing {file_path.name}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                total_lines += 1
                
                result = parse_gene_line(line, line_num)
                if result is None:
                    skipped_lines += 1
                    continue
                
                genome_id, organism, gene_id, gene_description = result
                
                prompt = construct_prompt(organism, gene_id, gene_description)
                
                prompts.append(prompt)
                genome_ids.append(genome_id)
                gene_ids.append(gene_id)
                organisms.append(organism)
    
    print_with_timestamp(f"Loaded {len(prompts)} prompts from {total_lines} total lines")
    if skipped_lines > 0:
        print_with_timestamp(f"Skipped {skipped_lines} invalid/empty lines")
    
    return prompts, genome_ids, gene_ids, organisms


def write_output(output_file, message: str) -> None:
    """Write message to output file or stdout.
    
    Parameters
    ----------
    output_file:
        File handle (or None for stdout).
    message:
        Message to write.
    """
    if output_file:
        output_file.write(message + "\n")
        output_file.flush()
    else:
        print(message)


async def process_with_scatter_gather(
    pool: VLLMPool,
    prompts: list[str],
    genome_ids: list[str],
    gene_ids: list[str],
    organisms: list[str],
    output_file,
    batch_size: int | None = None,
) -> tuple[int, int]:
    """Process prompts using scatter_gather pattern.
    
    Parameters
    ----------
    pool:
        VLLMPool instance connected to agent endpoints.
    prompts:
        List of prompts to process.
    genome_ids:
        List of genome IDs (parallel to prompts).
    gene_ids:
        List of gene IDs (parallel to prompts).
    organisms:
        List of organism names (parallel to prompts).
    output_file:
        Output file handle or None for stdout.
    batch_size:
        If specified, process prompts in batches of this size.
        If None, process all prompts at once.
    
    Returns
    -------
    tuple
        (successful_count, error_count)
    """
    success_count = 0
    error_count = 0
    
    if batch_size is None:
        batch_size = len(prompts)
    
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    
    print_with_timestamp(f"Processing {len(prompts)} prompts in {num_batches} batch(es) of up to {batch_size}")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(prompts))
        
        batch_prompts = prompts[start_idx:end_idx]
        batch_genome_ids = genome_ids[start_idx:end_idx]
        batch_gene_ids = gene_ids[start_idx:end_idx]
        batch_organisms = organisms[start_idx:end_idx]
        
        print_with_timestamp(
            f"Processing batch {batch_idx + 1}/{num_batches} "
            f"({len(batch_prompts)} prompts, indices {start_idx}-{end_idx-1})"
        )
        
        # Scatter prompts across agents and gather results
        responses = await scatter_gather(pool, batch_prompts)
        
        print_with_timestamp(f"Received {len(responses)} responses from batch {batch_idx + 1}")
        
        # Process responses in order
        for i, response in enumerate(responses):
            genome_id = batch_genome_ids[i]
            gene_id = batch_gene_ids[i]
            organism = batch_organisms[i]
            
            if not response.success or response.error:
                error_count += 1
                error_msg = response.error or "Unknown error"
                print_with_timestamp(
                    f"ERROR: Genome {genome_id}, Gene {gene_id}: {error_msg}"
                )
                write_output(output_file, f"[GENOME: {genome_id}] [ORGANISM: {organism}] [GENE: {gene_id}]")
                write_output(output_file, f"ERROR: {error_msg}")
                write_output(output_file, "\n" + "-" * 80 + "\n")
            else:
                success_count += 1
                write_output(output_file, f"[GENOME: {genome_id}] [ORGANISM: {organism}] [GENE: {gene_id}]")
                write_output(output_file, response.text)
                write_output(output_file, "\n" + "-" * 80 + "\n")
    
    return success_count, error_count


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        'input_dir',
        type=Path,
        help='Directory containing chunk_*.txt files with gene data',
    )
    parser.add_argument(
        '--hostfile',
        type=Path,
        help='Path to hostfile with agent endpoints (default: AURORA_SWARM_HOSTFILE env var)',
    )
    parser.add_argument(
        '--num-files',
        type=int,
        default=10,
        help='Number of chunk files to process (default: 10)',
    )
    parser.add_argument(
        '--skip-files',
        type=int,
        default=0,
        help='Number of chunk files to skip at the start (for resuming, default: 0)',
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file for results (default: stdout)',
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=1024,
        help='Maximum tokens per response (default: 1024)',
    )
    parser.add_argument(
        '--model',
        default='meta-llama/Llama-3.1-70B-Instruct',
        help='Model name (default: meta-llama/Llama-3.1-70B-Instruct)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Number of prompts per scatter_gather call (default: all at once)',
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=300.0,
        help='Per-request timeout in seconds (default: 300)',
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        default=64,
        help='Maximum concurrent requests (default: 64, reduce if seeing connection errors)',
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}", file=sys.stderr)
        return 1
    
    if not args.input_dir.is_dir():
        print(f"Error: Input path is not a directory: {args.input_dir}", file=sys.stderr)
        return 1
    
    # Determine hostfile path
    hostfile_path = args.hostfile
    if hostfile_path is None:
        hostfile_env = os.environ.get('AURORA_SWARM_HOSTFILE')
        if hostfile_env:
            hostfile_path = Path(hostfile_env)
        else:
            print(
                "Error: No hostfile specified. Use --hostfile or set AURORA_SWARM_HOSTFILE",
                file=sys.stderr,
            )
            print("\nExample hostfile format:", file=sys.stderr)
            print("  hostname1:8000", file=sys.stderr)
            print("  hostname2:8001", file=sys.stderr)
            print("  hostname3:8000 node=worker-01", file=sys.stderr)
            return 1
    
    if not hostfile_path.exists():
        print(f"Error: Hostfile not found: {hostfile_path}", file=sys.stderr)
        return 1
    
    print_with_timestamp("=" * 80)
    print_with_timestamp("Scatter-Gather COLI Example")
    print_with_timestamp("=" * 80)
    
    # Discover chunk files
    try:
        chunk_files = discover_chunk_files(args.input_dir, args.num_files, args.skip_files)
        if args.skip_files > 0:
            print_with_timestamp(f"Skipping first {args.skip_files} files")
        print_with_timestamp(f"Found {len(chunk_files)} chunk files to process")
        print_with_timestamp("=" * 80)
        print_with_timestamp("Files to process:")
        for i, f in enumerate(chunk_files, 1):
            file_num = args.skip_files + i
            print_with_timestamp(f"  {file_num:3d}. {f.name}")
        print_with_timestamp("=" * 80)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Read and prepare data
    prompts, genome_ids, gene_ids, organisms = read_and_prepare_data(chunk_files)
    
    if not prompts:
        print("Error: No valid prompts to process", file=sys.stderr)
        return 1
    
    # Parse hostfile and create pool
    print_with_timestamp(f"Loading endpoints from {hostfile_path}")
    endpoints = parse_hostfile(hostfile_path)
    print_with_timestamp(f"Loaded {len(endpoints)} endpoints")
    
    for i, ep in enumerate(endpoints[:5]):  # Show first 5
        print_with_timestamp(f"  Endpoint {i}: {ep.url}")
    if len(endpoints) > 5:
        print_with_timestamp(f"  ... and {len(endpoints) - 5} more")
    
    print_with_timestamp(f"Creating VLLMPool with model={args.model}, max_tokens={args.max_tokens}, concurrency={args.concurrency}")
    pool = VLLMPool(
        endpoints,
        model=args.model,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        concurrency=args.concurrency,
    )
    
    # Open output file if specified
    output_handle = None
    if args.output:
        output_handle = open(args.output, 'w', encoding='utf-8')
        print_with_timestamp(f"Writing output to {args.output}")
    
    try:
        # Process with scatter_gather
        print_with_timestamp("Starting scatter_gather processing...")
        start_time = datetime.now()
        
        success_count, error_count = await process_with_scatter_gather(
            pool,
            prompts,
            genome_ids,
            gene_ids,
            organisms,
            output_handle,
            batch_size=args.batch_size,
        )
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        # Print summary
        print_with_timestamp("=" * 80)
        print_with_timestamp("SUMMARY")
        print_with_timestamp("=" * 80)
        print_with_timestamp(f"Total prompts: {len(prompts)}")
        print_with_timestamp(f"Successful: {success_count}")
        print_with_timestamp(f"Errors: {error_count}")
        print_with_timestamp(f"Elapsed time: {elapsed:.2f} seconds")
        print_with_timestamp(f"Throughput: {len(prompts) / elapsed:.2f} prompts/second")
        print_with_timestamp("=" * 80)
        
    finally:
        # Clean up
        if output_handle:
            output_handle.close()
            print_with_timestamp(f"Output written to {args.output}")
        
        await pool.close()
        print_with_timestamp("Pool closed")
    
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
