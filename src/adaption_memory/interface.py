"""Common interface between memory systems and eval harnesses.

Every benchmark adapter converts its data into Sessions and drives a
MemorySystem through the same three calls, so the memory system under
test never sees benchmark-specific formats.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Turn:
    role: str  # "user" / "assistant", or a speaker name in human-human dialogs
    content: str


@dataclass
class Session:
    session_id: str
    date: str | None
    turns: list[Turn]


class MemorySystem(Protocol):
    def reset(self) -> None:
        """Start a fresh conversation history (new benchmark instance)."""
        ...

    def ingest(self, session: Session) -> None:
        """Consume one session at write time."""
        ...

    def answer(self, question: str, question_date: str | None = None,
               instruction: str | None = None) -> str:
        """Answer a question against everything ingested so far."""
        ...
