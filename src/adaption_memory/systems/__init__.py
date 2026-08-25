"""Memory systems under test.

Each system implements the MemorySystem protocol from
adaption_memory.interface and is constructed from a single answering LLM.
Register new systems in REGISTRY to make them available to the eval CLI
(`uv run eval <bench> --system <name>`) without touching the eval code.
"""

from collections.abc import Callable

from adaption_memory.interface import MemorySystem
from adaption_memory.llm import LLM
from adaption_memory.systems.adaptive import AdaptiveMemorySystem
from adaption_memory.systems.full_history import FullHistorySystem

REGISTRY: dict[str, Callable[[LLM], MemorySystem]] = {
    "adaptive": AdaptiveMemorySystem,
    "full-history": FullHistorySystem,
}
