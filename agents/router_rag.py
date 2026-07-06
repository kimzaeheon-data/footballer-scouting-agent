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

_REPORT_NO_HINTS = ["아니", "괜찮", "됐", "no", "말고", "필요없"]
_REPORT_YES_SHORT = {"응", "네", "어", "ㅇㅇ", "yes", "ok", "okay"}  # 첫 단어로만 인정 (부분일치 시 오탐 잦음)
_REPORT_YES_PHRASES = ["그래", "좋아", "부탁", "해줘", "해주세요"]  # 문장 어디에 있어도 인정

SYSTEM_PROMPT = """
너는 프리미어리그 2024-25 시즌 선수 데이터를 기반으로 답변하는 축구 스카우팅 어시스턴트야.

규칙:
1. 반드시 제공된 [선수 데이터]만 참고해서 답변해.
2. 데이터에 없는 정보(다른 리그, 다른 시즌, 연봉, 부상 등)는 "해당 정보는 데이터에 없습니다"라고 답해.
3. 선수 추천 시 근거 스탯을 명시해.
4. 한국어로 답변해.
5. [선수 데이터]로 제공된 선수들은 이미 질문 조건에 맞게 필터링/정렬까지 끝난 결과다. "카드 없이",
   "파울 없이"처럼 "없이/없는"이 들어간 질문이어도 이는 "0장/0회"라는 뜻이 아니라 "상대적으로 적은
   편"이라는 뜻이므로, 제공된 선수의 실제 수치가 0이 아니어도(예: 경고 카드 3장) 절대 "그런 선수는
   없습니다"라고 거절하지 마라 — 이미 조건에 맞게 걸러진 선수들이니 그대로 추천하고, 필요하면
   "카드가 아예 없진 않지만 상대적으로 적은 편"이라고 자연스럽게 설명해라.
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
        # nationality는 헤더 줄에 이미 넣으므로 stat_str에서는 제외 (중복 방지) — 이전엔
        # stat_str에서만 빠지고 헤더에도 없어서 국적 질문이 데이터가 있는데도 항상 거절당했음.
        lines = []
        for i, r in enumerate(results, 1):
            m = r["metadata"]
            stat_str = ", ".join(
                f"{k}: {v}" for k, v in m.items()
                if k not in ("player", "club", "position", "nationality")
            )
            lines.append(
                f"[선수 {i}] {m['player']} ({m.get('club', '?')}, {m.get('position', '?')}, "
                f"국적: {m.get('nationality', 'Unknown')})\n"
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
    def query(self, question: str, reset_history: bool = False, offer_report: bool = False) -> dict:
        """offer_report=True면 구조화 검색으로 선수가 나온 답변 끝에 스카우팅 리포트 제안을
        덧붙인다. 기본값 False — evaluate_router_full.py 등 기존 평가 스크립트의 answer_hit/
        faithfulness 채점에 영향을 주지 않기 위해 옵트인으로 둠 (실제 대화용 진입점에서만 켤 것)."""
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

        report_suggestion = None
        if offer_report and retrieval["mode"] == "structured" and retrieval["candidates"]:
            top_player = retrieval["candidates"][0]
            report_suggestion = f"{top_player} 선수 스카우팅 리포트를 이미지로 작성해드릴까요?"
            answer = f"{answer}\n\n{report_suggestion}"

        self.history.append({"role": "assistant", "content": answer})

        return {
            "question": question,
            "answer": answer,
            "retrieved": results,
            "retrieval_mode": retrieval["mode"],
            "report_suggestion": report_suggestion,
        }

    # ── 리포트 제안에 대한 사용자 응답 해석 (규칙 기반, LLM 호출 없음) ──
    @staticmethod
    def wants_report(reply: str) -> bool:
        """짧은 긍정어("어", "네" 등)는 다른 단어에 우연히 포함되기 쉬워서(예: "물어본거야"의 "어")
        문장 첫 단어로 나올 때만 인정한다. "부탁"/"해줘"처럼 덜 헷갈리는 표현은 어디 있어도 인정."""
        raw = (reply or "").strip()
        reply_lower = raw.lower()
        if any(h in reply_lower for h in _REPORT_NO_HINTS):
            return False
        tokens = reply_lower.replace("!", "").replace(".", "").replace(",", "").split()
        if tokens and tokens[0] in _REPORT_YES_SHORT:
            return True
        return any(h in reply_lower for h in _REPORT_YES_PHRASES)

    # ── 스카우팅 리포트 이미지 생성 ────────────────────────
    def generate_report(self, player_name: str, output_path: str = None) -> str:
        """tools/scouting_report.py로 이미지 카드를 만들어 저장된 파일 경로를 반환."""
        from tools.scouting_report import render_scouting_card
        return render_scouting_card(player_name, output_path=output_path)

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
