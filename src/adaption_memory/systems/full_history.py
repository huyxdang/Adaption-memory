"""Full-history baseline: no memory at all.

Keeps every ingested session verbatim and stuffs the whole history into the
prompt at answer time. This is the article's baseline column — expensive and
inaccurate on long histories, which is the point being demonstrated.
"""

from adaption_memory.interface import Session
from adaption_memory.llm import LLM

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to the user's past conversation "
    "sessions, shown with their timestamps. Answer the question using only "
    "that history. Be concise. If the history does not contain the "
    "information needed, say that the information is not available."
)


class FullHistorySystem:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.sessions: list[Session] = []

    def reset(self) -> None:
        self.sessions = []

    def ingest(self, session: Session) -> None:
        self.sessions.append(session)

    def render_history(self) -> str:
        blocks = []
        for s in self.sessions:
            header = f"[Session {s.session_id}" + (f" — {s.date}]" if s.date else "]")
            lines = [f"{t.role.upper()}: {t.content}" for t in s.turns]
            blocks.append(header + "\n" + "\n".join(lines))
        return "\n\n".join(blocks)

    def answer(self, question: str, question_date: str | None = None,
               instruction: str | None = None) -> str:
        parts = ["<history>", self.render_history(), "</history>", ""]
        if question_date:
            parts.append(f"Current date: {question_date}")
        parts.append(f"Question: {question}")
        if instruction:
            parts.append(instruction)
        return self.llm.chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(parts)},
        ])
