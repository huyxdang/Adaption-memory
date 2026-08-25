"""End-to-end write-time memory system composition."""

from __future__ import annotations

from pathlib import Path

from adaption_memory.interface import Session
from adaption_memory.llm import LLM
from adaption_memory.evals.common import usage_delta
from adaption_memory.memory.answer import MemoryAnswerer
from adaption_memory.memory.checkpoint import Checkpoint, replay_usage, stable_hash
from adaption_memory.memory.embedding import LocalEmbedder
from adaption_memory.memory.extract import ExtractionResult, MemoryExtractor
from adaption_memory.memory.retrieve import HybridRetriever, Retrieved
from adaption_memory.memory.store import MemoryStore


class WriteTimeMemorySystem:
    def __init__(self, *, extractor_llm: LLM, answer_llm: LLM,
                 store_path: str | Path, checkpoint_dir: str | Path,
                 arm: str, fewshot: bool, format_name: str = "F1",
                 prompt_revision: str = "base", emission: str = "pointer",
                 local_thinking: bool = False, answer_revision: str = "base",
                 retrieval_k: int = 12, dense_weight: float = 1.0,
                 bm25_weight: float = 1.0, demotion_factor: float = 0.3,
                 adaptive_k: bool = False, multi_query: bool = False):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.store = MemoryStore(store_path)
        self.embedder = LocalEmbedder()
        self.extractor_llm = extractor_llm
        self.answer_llm = answer_llm
        self.extractor = MemoryExtractor(
            extractor_llm, self.store, self.embedder,
            checkpoint_dir / "extractions.jsonl", fewshot=fewshot,
            format_name=format_name, arm=arm,
            prompt_revision=prompt_revision, emission=emission,
            local_thinking=local_thinking,
        )
        self.retriever = HybridRetriever(
            self.store, self.embedder,
            checkpoint_dir / "retrievals.jsonl", k=retrieval_k,
            dense_weight=dense_weight, bm25_weight=bm25_weight,
            demotion_factor=demotion_factor, format_name=format_name,
            adaptive_k=adaptive_k,
        )
        self.answerer = MemoryAnswerer(
            answer_llm, checkpoint_dir / "answer_calls.jsonl", arm,
            format_name=format_name, answer_revision=answer_revision,
        )
        self.multi_query = multi_query
        self.expansion_checkpoint = (
            Checkpoint(checkpoint_dir / "expansions.jsonl")
            if multi_query else None
        )
        self.extractions: list[ExtractionResult] = []
        self.last_retrieved: list[Retrieved] = []
        self.last_answer_hash: str | None = None

    def reset(self) -> None:
        """The state is isolated by store path; no destructive reset exists."""

    def close(self) -> None:
        self.store.close()

    def usage(self) -> dict:
        left = self.extractor_llm.usage.snapshot()
        right = self.answer_llm.usage.snapshot()
        return {key: left.get(key, 0) + right.get(key, 0)
                for key in left.keys() | right.keys()}

    def ingest(self, session: Session) -> None:
        self.extractions.append(self.extractor.extract(session))

    def _expand_queries(self, question: str) -> list[str]:
        """Two Luna-written paraphrases beside the original question, so
        vocabulary mismatches between question and stored phrasing still
        land a retrieval hit. Cached like every other model call."""
        input_hash = stable_hash({"schema": 1, "task": "query-expansion",
                                  "model": self.answer_llm.model,
                                  "question": question})
        cached = self.expansion_checkpoint.get(input_hash)
        if cached is not None:
            replay_usage(self.answer_llm.usage, cached.get("usage"))
            return cached["queries"]
        usage_before = self.answer_llm.usage.snapshot()
        raw = self.answer_llm.chat([
            {"role": "system", "content":
                "Rewrite the question as two alternative search queries that "
                "use different wording for the same information need. Output "
                "exactly two lines, one query per line, nothing else."},
            {"role": "user", "content": question},
        ], max_tokens=120)
        queries = [question] + [line.strip() for line in raw.splitlines()
                                if line.strip()][:2]
        self.expansion_checkpoint.append({
            "input_hash": input_hash, "question": question,
            "queries": queries,
            "usage": usage_delta(usage_before,
                                 self.answer_llm.usage.snapshot()),
        })
        return queries

    def answer(self, question: str, question_date: str | None = None,
               instruction: str | None = None) -> str:
        if self.multi_query:
            merged: dict[str, object] = {}
            for query in self._expand_queries(question):
                for item in self.retriever.retrieve(query):
                    kept = merged.get(item.record.id)
                    if kept is None or item.score > kept.score:
                        merged[item.record.id] = item
            ranked = sorted(merged.values(), key=lambda item: -item.score)
            self.last_retrieved = ranked[:self.retriever.k]
            answer, input_hash = self.answerer.answer(
                question, self.last_retrieved, question_date, instruction
            )
            self.last_answer_hash = input_hash
            return answer
        self.last_retrieved = self.retriever.retrieve(question)
        answer, input_hash = self.answerer.answer(
            question, self.last_retrieved, question_date, instruction
        )
        self.last_answer_hash = input_hash
        return answer
