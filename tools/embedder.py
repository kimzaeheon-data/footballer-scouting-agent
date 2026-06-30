"""
임베딩 모델 클래스
- BGEEmbedder  : BAAI/bge-m3 (HuggingFace, 로컬)
- OpenAIEmbedder: text-embedding-3-small (OpenAI API)
"""

from abc import ABC, abstractmethod
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """텍스트 리스트 → numpy 배열 (n, dim)"""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """단일 쿼리 → numpy 배열 (dim,)"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


# ── BGE-M3 (HuggingFace) ──────────────────────────────────
class BGEEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "BAAI/bge-m3", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer
        print(f"[BGEEmbedder] 모델 로딩: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self._name = "bge-m3"

    @property
    def name(self) -> str:
        return self._name

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return np.array(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vec = self.model.encode(
            text,
            normalize_embeddings=True,
        )
        return np.array(vec, dtype=np.float32)


# ── OpenAI Embedder ───────────────────────────────────────
class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "text-embedding-3-small", batch_size: int = 100):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.batch_size = batch_size
        self._name = "openai-text-embedding-3-small"

    @property
    def name(self) -> str:
        return self._name

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed(self, texts: list[str]) -> np.ndarray:
        all_vecs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vecs = self._call_api(batch)
            all_vecs.extend(vecs)
            print(f"[OpenAIEmbedder] {min(i + self.batch_size, len(texts))}/{len(texts)} 완료")
        return np.array(all_vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vecs = self._call_api([text])
        return np.array(vecs[0], dtype=np.float32)
