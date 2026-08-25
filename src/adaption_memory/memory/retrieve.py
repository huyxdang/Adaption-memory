"""Hybrid dense/BM25 retrieval with RRF and supersession handling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import regex
from rank_bm25 import BM25Okapi

from adaption_memory.memory.checkpoint import Checkpoint, stable_hash
from adaption_memory.memory.embedding import LocalEmbedder
from adaption_memory.memory.store import MemoryStore, Record

AGGREGATION_CUES = regex.compile(
    r"\b(how many|how much|how often|total|altogether|in all|count|"
    r"sum|combined|overall|number of|times did|times has)\b", regex.I)
AGGREGATION_K = 24


@dataclass(frozen=True)
class Retrieved:
    record: Record
    score: float
    dense_rank: int | None
    bm25_rank: int | None

    def as_dict(self) -> dict:
        return {
            **self.record.as_dict(),
            "score": self.score,
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
        }


class HybridRetriever:
    def __init__(self, store: MemoryStore, embedder: LocalEmbedder,
                 checkpoint_path: str | Path, *, k: int = 12,
                 dense_weight: float = 1.0, bm25_weight: float = 1.0,
                 demotion_factor: float = 0.3, format_name: str = "F1",
                 adaptive_k: bool = False):
        if not 1 <= k <= 64:
            raise ValueError("retrieval k must be between 1 and 64")
        self.adaptive_k = adaptive_k
        self.store = store
        self.embedder = embedder
        self.checkpoint = Checkpoint(checkpoint_path)
        self.k = k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.demotion_factor = demotion_factor
        self.format_name = format_name

    def retrieve(self, query: str) -> list[Retrieved]:
        # Aggregation questions ("how many…", "total…") need every matching
        # piece in context, not just the top-k most similar; blanket k=20
        # measured worse on ordinary questions, so widen only on cue.
        k = self.k
        if self.adaptive_k and AGGREGATION_CUES.search(query):
            k = max(self.k, AGGREGATION_K)
        records = self.store.all()
        state_hash = stable_hash([record.as_dict() for record in records])
        input_hash = stable_hash({
            "schema": 1, "query": query, "state_hash": state_hash,
            "k": k, "dense_weight": self.dense_weight,
            "bm25_weight": self.bm25_weight,
            "demotion_factor": self.demotion_factor,
            "format": self.format_name,
        })
        cached = self.checkpoint.get(input_hash)
        if cached is not None:
            out = []
            for row in cached["records"]:
                record = self.store.get(row["id"])
                if record is not None:
                    out.append(Retrieved(record, row["score"],
                                         row.get("dense_rank"),
                                         row.get("bm25_rank")))
            return out
        if not records:
            self.checkpoint.append({
                "input_hash": input_hash, "query": query,
                "state_hash": state_hash, "records": [],
            })
            return []

        query_vector = self.embedder.encode_one(query)
        dense_scores = []
        for record in records:
            if record.embedding is None:
                score = -1.0
            else:
                denom = np.linalg.norm(query_vector) * np.linalg.norm(record.embedding)
                score = float(np.dot(query_vector, record.embedding) / denom) if denom else 0.0
            dense_scores.append(score)
        tokenized = [self._tokens(record.search_text()) for record in records]
        bm25 = BM25Okapi(tokenized)
        bm25_scores = list(bm25.get_scores(self._tokens(query)))
        dense_order = sorted(range(len(records)), key=lambda i: dense_scores[i], reverse=True)
        bm25_order = sorted(range(len(records)), key=lambda i: bm25_scores[i], reverse=True)
        dense_rank = {index: rank + 1 for rank, index in enumerate(dense_order)}
        bm25_rank = {index: rank + 1 for rank, index in enumerate(bm25_order)}

        fused = []
        for index, record in enumerate(records):
            score = (
                self.dense_weight / (60 + dense_rank[index])
                + self.bm25_weight / (60 + bm25_rank[index])
            )
            if self.store.is_superseded(record.id):
                score *= self.demotion_factor
            fused.append(Retrieved(record, score, dense_rank[index], bm25_rank[index]))
        fused.sort(key=lambda item: item.score, reverse=True)

        selected = fused[:k]
        by_id = {item.record.id: item for item in fused}
        expanded = list(selected)
        selected_ids = {item.record.id for item in expanded}
        for item in list(selected):
            for successor in self.store.successor_chain(item.record.id):
                if successor.id not in selected_ids:
                    expanded.append(by_id[successor.id])
                    selected_ids.add(successor.id)
        expanded.sort(key=lambda item: item.score, reverse=True)

        if self.format_name == "F3":
            entity_ids = {
                entity.casefold()
                for item in expanded[:k]
                for entity in item.record.entities
            }
            for item in fused:
                if (item.record.id not in selected_ids
                        and any(entity.casefold() in entity_ids
                                for entity in item.record.entities)
                        and not self.store.is_superseded(item.record.id)):
                    expanded.append(item)
                    selected_ids.add(item.record.id)
            # Entity dossiers are bounded to avoid unbounded context growth.
            expanded = expanded[:max(self.k, 24)]
        else:
            expanded = expanded[:k]

        self.checkpoint.append({
            "input_hash": input_hash, "query": query,
            "state_hash": state_hash,
            "records": [item.as_dict() for item in expanded],
        })
        return expanded

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return regex.findall(r"[\p{L}\p{N}]+", text.casefold())
