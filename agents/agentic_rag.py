"""
Agentic RAG 파이프라인 (LangGraph ReAct 에이전트)

naive_rag → router_rag(고정 스크립트가 "구조화 vs 벡터"를 라우팅) → agentic_rag(에이전트가
스스로 어떤 툴을, 몇 번, 어떤 인자로 호출할지 결정) 순으로 자율성이 늘어난다.

router_rag의 StructuredQueryParser(질문→스펙 변환 전용 LLM 호출)가 여기서는 사라진다.
그 역할을 에이전트의 tool-calling 자체가 대신하기 때문이다 — 이게 "agentic"의 핵심 차이.
후속 질문("그 중에서 ~") 처리도 router_rag처럼 별도 last_candidates 상태값으로 관리하지 않고,
에이전트가 대화 메시지 히스토리를 직접 읽고 stat_query의 player_names 인자에 이전 결과를
스스로 채워 넣는 방식으로 처리한다.
"""

import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from tools.retriever import FAISSRetriever
from tools.agentic_tools import ALL_TOOLS, set_vector_retriever

load_dotenv()

SYSTEM_PROMPT = """
너는 프리미어리그 2024-25 시즌 선수 데이터를 기반으로 답변하는 축구 스카우팅 어시스턴트야.
아래 툴을 사용해서 실제 데이터를 확인한 뒤에만 답변해.

[툴 선택 원칙]
1. 최상급 표현("가장 많은/높은/적은")이 없어도, "A 이상이면서 B도 높은", "A와 B가 모두 높은"처럼
   실질적으로 포지션+스탯 조건으로 선수를 걸러내거나 순위를 매기는 질문이면 전부 stat_query를 써.
   vector_search는 "~스타일에 맞는", "~같은 느낌의 선수"처럼 정형화된 컬럼 조건으로 도저히
   못 바꾸는 서술형 질문에만 최후의 수단으로 사용해. 애매하면 stat_query를 우선 시도해라.
2. 질문에 선수 이름이 실제 텍스트로 등장할 때만("메수트 외질 스탯 알려줘"처럼) player_lookup을 써.
   "그 선수"처럼 대명사로만 지칭하는 경우는 이름이 아니므로 player_lookup을 쓰지 마라 — 3번을 따라라.
3. 후속 질문에서 "그 중에서"뿐 아니라 "그 선수"처럼 대명사로 직전 후보를 다시 가리킬 때도 전부
   stat_query를 써. 직전 결과가 여러 명이었어도 마찬가지다: 후보 이름들을 전부 stat_query의
   player_names 인자로 넘기고 top_n으로 몇 명으로 좁힐지 정해라 (position은 이 경우 무시됨).
   "누군지 확인해본다"며 후보들을 player_lookup으로 한 명씩 따로 조회하지 마라 — stat_query
   한 번으로 끝내는 게 원칙이다 (4번 참고).
4. 한 번 호출해서 합리적인 결과를 얻었으면 같은 질문에 stat_query를 반복 호출하지 마. 여러 번
   부르면 그중 정확히 어떤 해석이 맞는지 너 스스로도 헷갈리게 되고 답변이 부정확해진다. 인자를
   신중하게 한 번에 정하는 게 여러 번 시도해보는 것보다 낫다.

[stat_query 인자를 정확히 채우는 법 — 여기서 실수가 제일 많이 남]
- 복합조건 질문("A 이상이면서 B도 많은 X", "A와 B가 모두 높은 X"):
  명시적 숫자가 있는 조건은 filters에 절대값(>=,<=,==,>,<)으로. 숫자가 없는 조건 중, 문장에서
  나중에 언급되거나 "~도 많은/높은/좋은"으로 강조된 쪽을 metric(정렬 기준)으로 삼는다.
  "A와 B가 모두 높은" 처럼 대등하게 나열된 경우도 예외 없이: 먼저 언급된 A는 percentile filter
  (op=">=pct", value=50), 나중에 언급된 B는 metric. metric으로 쓴 컬럼은 filters에 다시 넣지 마라.
  예) "태클과 인터셉트가 모두 높은 수비수는?" → filters=[Tackles>=pct50], metric=Interceptions.
  예) "패스 성공률 80% 이상이면서 태클도 많은 미드필더는?" → filters=[Passes%>=80], metric=Tackles.
  예) "블록과 클리어런스 모두 많은 수비수는?" → filters=[Blocks>=pct50], metric=Clearances.
  (사이에 "헌신적인/열심히 뛰는" 같은 수식어가 끼어 있어도 순서 규칙은 그대로 적용 — 수식어는
  filter/metric 판단에 영향을 주지 않는다.)
- 주의: "카드 없이", "파울 없이"처럼 "없이/없는"이 나와도 절대 op="=="/value=0으로 쓰지 마라.
  "적은 편"이라는 뜻이지 완전히 0이라는 뜻이 아니다. 반드시 op="<=pct", value=50을 써라.
- "주전/정규 선발/풀타임"은 명시 숫자가 없으면 Minutes>=1800을 기본으로 써라.
- 포지션 매핑: 공격수/스트라이커/윙어/포워드→FWD, 미드필더→MID, 수비수/센터백/풀백→DEF, 골키퍼→GKP.
- top_n: 특정 인원수가 명시 안 된 새 순위 질문은 무조건 top_n=3 (이 데이터셋에서는 추천 질문에
  보통 3명을 기대함, 문법이 단수라고 top_n=1로 줄이지 마라). 후속 질문에서 "그 선수는?"처럼
  단수로 좁힐 때만 top_n=1을 써라. top_n을 필요 이상으로 크게 잡지 마라 (정확도가 떨어진다).
  **"인터셉트가 가장 많은 미드필더는?"처럼 "누구는?"으로 묻는 가장 단순한 단일 최상급 질문도
  예외 없이 top_n=3이다.** "가장 많은"이 문법적으로 1명을 묻는 것처럼 들려도, 이 질문 유형은
  전부 3명을 반환해야 한다 — top_n=1은 오직 use_previous_candidates 후속 질문에서만 쓴다.
- 후속 질문 처리: "그 중 X가 많은/적은 선수는?"(그룹 내 재선별)은 X를 metric으로 쓰고
  sort_direction을 맞춰라(많은/높은→desc, 적은/낮은→asc), filters는 비워둬라. 이때 top_n은 반드시
  1이다 — "그 선수는?"이라고 "그"를 붙이지 않고 그냥 "~한 선수는?"으로만 물어도, "그 중"으로
  그룹을 전제한 뒤 특정 기준으로 되묻는 질문은 "그 중에서 그 기준에 제일 맞는 한 명"을 찾으라는
  뜻이다. top_n=3을 쓰는 건 오직 새 질문(use_previous_candidates=false)일 때뿐이다.
  예) "그 중 프로그레시브 캐리도 많은 선수는?" → metric=Progressive Carries, top_n=1 (3명 아님).
  반면 "그 선수 ~은?"(단수로 이미 정해진 1명을 다시 가리키며 설명·계산을 요구, 특히 목록에 없는
  파생 지표를 물을 때)은 metric="NONE"으로 두고 player_names로 이전 후보를 그대로 유지한 채
  top_n=1만 적용해 — 새로 재정렬하지 마라.
  **직전 턴 결과가 이미 여러 명(2명 이상)이었는데 이번 턴이 "그 선수"처럼 단수로 되짚는 경우,
  "그 선수"는 직전 결과 중 1등(가장 먼저 나온/가장 순위가 높은 후보) 한 명을 가리킨다.** 이럴 땐
  직전 후보 전체를 player_names로 넘기고 top_n=1을 적용해 자동으로 1등만 추리면 된다 (metric은
  직전 정렬 기준을 유지하거나, 파생 지표를 묻는 거면 NONE). 후보가 누구인지 확인한다고 각 선수를
  따로따로 여러 번 조회하거나 stat_query를 반복 호출하지 마라 — 위 규칙 4에서 말했듯 한 번에
  정확히 정하는 게 원칙이다.

답변 규칙:
1. 반드시 툴 조회 결과(실제 데이터)만 근거로 답변해. 툴로 확인 안 된 내용을 지어내지 마.
2. 다른 리그/시즌/연봉/부상 등 이 데이터셋에 없는 정보를 물으면, 툴을 호출해도 관련 데이터가
   안 나오면 "해당 정보는 데이터에 없습니다"라고 정직하게 답해.
3. 선수 추천 시 근거 스탯 수치를 명시해.
4. 한국어로 답변해.
""".strip()


class AgenticRAG:
    def __init__(
        self,
        vector_retriever: FAISSRetriever,
        model: str = "gpt-4o-mini",
    ):
        set_vector_retriever(vector_retriever)
        self.llm = ChatOpenAI(model=model, temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        self.agent = create_react_agent(self.llm, ALL_TOOLS, prompt=SYSTEM_PROMPT)
        self.messages: list = []

    # ── 단일 쿼리 ────────────────────────────────────────
    def query(self, question: str, reset_history: bool = False) -> dict:
        if reset_history:
            self.messages = []

        self.messages.append(HumanMessage(content=question))
        turn_start = len(self.messages) - 1  # 방금 넣은 HumanMessage의 인덱스

        result = self.agent.invoke({"messages": self.messages})
        self.messages = result["messages"]

        new_messages = self.messages[turn_start + 1:]
        answer = new_messages[-1].content if new_messages else ""

        # 이번 turn에서 호출된 툴 이름들 + 각 호출의 player 리스트를 순서대로 모아둔다.
        # 에이전트는 종종 stat_query를 여러 번 호출하며 해석을 바꿔가는데, 최종 답변은
        # 보통 "가장 마지막" 툴 호출 결과를 근거로 쓴다. 그래서:
        #   - retrieved_players : 마지막 툴 호출의 player 리스트 (최종 답변 근거에 가장 가까움 → 평가용)
        #   - all_tool_players  : 모든 호출을 순서대로 합친 리스트 (디버깅/투명성용, 이전 동작과 동일)
        tools_used: list[str] = []
        per_call_players: list[list[str]] = []
        per_call_detail: list[list[dict]] = []
        all_tool_players: list[str] = []
        for m in new_messages:
            if isinstance(m, ToolMessage):
                tools_used.append(m.name)
                try:
                    payload = json.loads(m.content)
                    players = payload.get("players", [])
                    detail = payload.get("detail", [])
                except (json.JSONDecodeError, TypeError):
                    players = []
                    detail = []
                per_call_players.append(players)
                per_call_detail.append(detail)
                for p in players:
                    if p not in all_tool_players:
                        all_tool_players.append(p)

        retrieved_players = per_call_players[-1] if per_call_players else []
        retrieved_detail = per_call_detail[-1] if per_call_detail else []

        return {
            "question": question,
            "answer": answer,
            "retrieved_players": retrieved_players,
            "retrieved_detail": retrieved_detail,
            "all_tool_players": all_tool_players,
            "tools_used": tools_used,
        }

    def reset(self):
        self.messages = []


# ── 간단 테스트 ──────────────────────────────────────────
if __name__ == "__main__":
    from tools.embedder import OpenAIEmbedder

    embedder = OpenAIEmbedder()
    vec_retriever = FAISSRetriever(embedder).build()
    rag = AgenticRAG(vec_retriever)

    print("=== Agentic RAG 테스트 ===\n")
    q1 = "태클 많은 미드필더 3명 추천해줘"
    r1 = rag.query(q1)
    print(f"Q: {q1}\n[tools: {r1['tools_used']}]\nA: {r1['answer']}\n")

    q2 = "그 중에서 패스 성공률이 제일 높은 선수는?"
    r2 = rag.query(q2)
    print(f"Q: {q2}\n[tools: {r2['tools_used']}]\nA: {r2['answer']}\n")
