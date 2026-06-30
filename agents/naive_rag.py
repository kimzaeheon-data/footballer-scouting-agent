"""
Naive RAG 파이프라인
Query → FAISS 검색 → GPT-4o 답변 생성
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from tools.retriever import FAISSRetriever

load_dotenv()

SYSTEM_PROMPT = """
너는 프리미어리그 2024-25 시즌 선수 데이터를 기반으로 답변하는 축구 스카우팅 어시스턴트야.

규칙:
1. 반드시 제공된 [선수 데이터]만 참고해서 답변해.
2. 데이터에 없는 정보(다른 리그, 다른 시즌, 연봉, 부상 등)는 "해당 정보는 데이터에 없습니다"라고 답해.
3. 선수 추천 시 근거 스탯을 명시해.
4. 한국어로 답변해.
""".strip()


class NaiveRAG:
    def __init__(self, retriever: FAISSRetriever, top_k: int = 5, model: str = "gpt-4o"):
        self.retriever = retriever
        self.top_k = top_k
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.history: list[dict] = []  # 대화 히스토리 (followup 지원)

    # ── 컨텍스트 빌드 ────────────────────────────────────
    def _build_context(self, results: list[dict]) -> str:
        lines = []
        for i, r in enumerate(results, 1):
            m = r["metadata"]
            lines.append(
                f"[선수 {i}] {m['player']} ({m['club']}, {m['position']})\n"
                f"  출전: {m['minutes']}분 | 골: {m['goals']} | 어시스트: {m['assists']}\n"
                f"  패스 성공률: {m['passes_pct']}% | 태클: {m['tackles']} | "
                f"인터셉트: {m['interceptions']} | 상대 진영 전진: {m['progressive_carries']}회\n"
                f"  유사도 점수: {r['score']:.4f}"
            )
        return "\n\n".join(lines)

    # ── 단일 쿼리 ────────────────────────────────────────
    def query(self, question: str, reset_history: bool = False) -> dict:
        if reset_history:
            self.history = []

        # 검색
        results = self.retriever.search(question, top_k=self.top_k)
        context = self._build_context(results)

        # 메시지 구성
        user_message = f"[선수 데이터]\n{context}\n\n[질문]\n{question}"
        self.history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

        # LLM 호출
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
        )
        answer = response.choices[0].message.content

        self.history.append({"role": "assistant", "content": answer})

        return {
            "question": question,
            "answer": answer,
            "retrieved": results,
            "embedder": self.retriever.embedder.name,
        }

    def reset(self):
        self.history = []


# ── 간단 테스트 ──────────────────────────────────────────
if __name__ == "__main__":
    from tools.embedder import BGEEmbedder, OpenAIEmbedder

    print("=== Naive RAG 테스트 ===\n")
    question = "태클과 인터셉트가 많은 미드필더 추천해줘"

    for EmbedderClass in [BGEEmbedder, OpenAIEmbedder]:
        embedder = EmbedderClass()
        retriever = FAISSRetriever(embedder).build(force=True)
        rag = NaiveRAG(retriever)

        result = rag.query(question)
        print(f"\n[{result['embedder']}]")
        print(f"Q: {result['question']}")
        print(f"A: {result['answer']}\n")
        print("-" * 60)
