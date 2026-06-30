"""
FAISS 기반 벡터 검색기
- FAISSRetriever: 청크 임베딩 → 인덱스 빌드 → 유사도 검색
"""

import json
import os
import pickle
import numpy as np
import faiss
from tools.embedder import BaseEmbedder

CHUNKS_PATH = "data/processed/players_chunked.json"
INDEX_DIR = "data/index"


class FAISSRetriever:
    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder
        self.index = None
        self.chunks = []
        self.index_path = os.path.join(INDEX_DIR, f"{embedder.name}.faiss")
        self.chunks_path = os.path.join(INDEX_DIR, f"{embedder.name}_chunks.pkl")

    # ── 인덱스 빌드 ──────────────────────────────────────
    def build(self, chunks_path: str = CHUNKS_PATH, force: bool = False):
        os.makedirs(INDEX_DIR, exist_ok=True)

        # 이미 인덱스 있으면 로드
        if not force and os.path.exists(self.index_path):
            print(f"[{self.embedder.name}] 기존 인덱스 로드: {self.index_path}")
            self._load()
            return self

        # 청크 로드
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        texts = [c["text"] for c in self.chunks]
        print(f"[{self.embedder.name}] {len(texts)}개 청크 임베딩 시작...")

        vecs = self.embedder.embed(texts)  # (n, dim)
        dim = vecs.shape[1]

        # Inner Product 인덱스 (정규화된 벡터 → cosine similarity)
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vecs)

        # 저장
        faiss.write_index(self.index, self.index_path)
        with open(self.chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

        print(f"[{self.embedder.name}] 인덱스 저장 완료 ({self.index.ntotal}개 벡터)")
        return self

    # ── 인덱스 로드 ──────────────────────────────────────
    def _load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

    # ── 검색 ─────────────────────────────────────────────
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Returns:
            list of {"text": ..., "metadata": ..., "score": ...}
        """
        if self.index is None:
            raise RuntimeError("인덱스가 없습니다. build()를 먼저 실행하세요.")

        q_vec = self.embedder.embed_query(query).reshape(1, -1)
        scores, indices = self.index.search(q_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "score": float(score),
            })
        return results
