"""
Router RAG 파이프라인 (naive RAG → agentic RAG로 가는 다리)
Query → [구조화 라우터] → is_stat_query?
    True  → StructuredRetriever (pandas filter/sort, 정확한 top-N)
    False → FAISSRetriever (기존 벡터 검색, 서술형/거절 케이스 폴백)
    → GPT-4o 답변 생성

naive_rag.py와 동일한 SYSTEM_PROMPT/답변 스타일을 유지하되, 검색 단계만 라우팅한다.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from tools.retriever import FAISSRetriever
from tools.structured_query import StructuredQueryParser, StructuredRetriever

load_dotenv()

SYSTEM_PROMPT = """
너는 프리미어리그 2024-25 시즌 선수 데이터를 기반으로 답변하는 축구 스카우팅 어시스턴트야.

규칙:
1. 반드시 제공된 [선수 데이터]만 참고해서 답변해.
2. 데이터에 없는 정보(다른 리그, 다른 시즌, 연봉, 부상 등)는 "해당 정보는 데이터에 없습니다"라고 답해.
3. 선수 추천 시 근거 스탯을 명시해.
4. 한국어로 답변해.
""".strip()


class RouterRAG:
    def __init__(
        self,
        vector_retriever: FAISSRetriever,
        top_k: int = 5,
        model: str = "gpt-4o",
        parser_model: str = "gpt-4o-mini",
    ):
        self.vector_retriever = vector_retriever
        self.parser = StructuredQueryParser(model=parser_model)
        self.struct_retriever = StructuredRetriever()
        self.top_k = top_k
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

        self.history: list[dict] = []
        self.last_candidates: list[str] = []

    # ── 컨텍스트 빌드 ────────────────────────────────────
    def _build_context(self, results: list[dict]) -> str:
        lines = []
        for i, r in enumerate(results, 1):
            m = r["metadata"]
            stat_str = ", ".join(
                f"{k}: {v}" for k, v in m.items()
                if k not in ("player", "club", "position", "nationality")
            )
            lines.append(
                f"[선수 {i}] {m['player']} ({m.get('club', '?')}, {m.get('position', '?')})\n"
                f"  {stat_str}"
            )
        return "\n\n".join(lines)

    def _history_context(self) -> str:
        if not self.last_candidates:
            return ""
        return f"직전 turn의 후보 선수: {', '.join(self.last_candidates)}"

    # ── 검색만 수행 (평가용, LLM 답변 생성 없음) ──────────
    def retrieve(self, question: str, reset: bool = False) -> dict:
        if reset:
            self.last_candidates = []

        spec = self.parser.parse(question, history_context=self._history_context())

        # use_previous_candidates가 true면 후보 carryover 자체가 목적이므로
        # (예: "그 선수 소속팀이 어디야?"처럼 랭킹 질문이 아닌 서술형 후속 질문도 포함)
        # is_stat_query 여부와 무관하게 구조화 경로를 탄다. 새 질문일 때만 is_stat_query +
        # metric이 있어야 구조화 경로를 쓴다.
        can_use_structured = (
            spec.get("use_previous_candidates") and bool(self.last_candidates)
        ) or (
            spec.get("is_stat_query") and spec.get("metric", "NONE") != "NONE"
        )

        if can_use_structured:
            prev = self.last_candidates if spec.get("use_previous_candidates") else None
            results, candidates = self.struct_retriever.execute(spec, previous_candidates=prev)
            mode = "structured"
        else:
            results = self.vector_retriever.search(question, top_k=self.top_k)
            candidates = [r["metadata"]["player"] for r in results]
            mode = "vector"

        if candidates:
            self.last_candidates = candidates

        return {"results": results, "candidates": candidates, "mode": mode, "spec": spec}

    # ── 단일 쿼리 (검색 + 답변 생성) ───────────────────────
    def query(self, question: str, reset_history: bool = False) -> dict:
        if reset_history:
            self.history = []
            self.last_candidates = []

        retrieval = self.retrieve(question, reset=False)
        results = retrieval["results"]
        context = self._build_context(results)

        user_message = f"[선수 데이터]\n{context}\n\n[질문]\n{question}"
        self.history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

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
            "retrieval_mode": retrieval["mode"],
        }

    def reset(self):
        self.history = []
        self.last_candidates = []


# ── 간단 테스트 ──────────────────────────────────────────
if __name__ == "__main__":
    from tools.embedder import OpenAIEmbedder

    embedder = OpenAIEmbedder()
    vec_retriever = FAISSRetriever(embedder).build()
    rag = RouterRAG(vec_retriever)

    print("=== Router RAG 테스트 ===\n")
    q1 = "태클 많은 미드필더 3명 추천해줘"
    r1 = rag.query(q1)
    print(f"Q: {q1}\n[{r1['retrieval_mode']}]\nA: {r1['answer']}\n")

    q2 = "그 중에서 패스 성공률이 제일 높은 선수는?"
    r2 = rag.query(q2)
    print(f"Q: {q2}\n[{r2['retrieval_mode']}]\nA: {r2['answer']}\n")
