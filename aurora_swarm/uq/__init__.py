"""Lab 3: Semantic uncertainty for LLM-based biological reasoning.

Provides cluster-based semantic entropy, kernel language entropy (KLE),
and semantic entropy probes. Use with Aurora-Swarm patterns (broadcast,
scatter-gather) for LLM sampling and embedding.
"""

from aurora_swarm.uq.semantic_entropy import semantic_entropy
from aurora_swarm.uq.kle import kernel_language_entropy
from aurora_swarm.uq.probes import train_probe, predict_semantic_entropy, save_probe, load_probe

__all__ = [
    "semantic_entropy",
    "kernel_language_entropy",
    "train_probe",
    "predict_semantic_entropy",
    "save_probe",
    "load_probe",
]
