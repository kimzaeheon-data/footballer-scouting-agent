"""
Cascade RAG 파이프라인 (router_rag → agentic_rag 에스컬레이션)

RAG_COMPARISON_REPORT.md 결론: router_rag가 정확도(98.55%)·결정론성 둘 다 앞서 1순위 권장.
agentic_rag(97.10%)는 player_lookup 툴과 자율적 재해석 능력이 있지만 더 비싸고 비결정적.

그럼 agentic_rag는 왜 남겨두는가? router_rag의 StructuredQueryParser가 "이 질문을 스탯
필터/정렬로 못 알아듣겠다"는 신호를 스스로 낼 때가 있다 (예: is_stat_query=True인데 metric도
filters도 못 뽑음, 또는 구조화 검색이 결과를 하나도 못 찾음). 이런 저신뢰 케이스에서만
agentic_rag로 에스컬레이션해서 회복을 시도한다 — "싸고 빠른 경로를 기본으로 쓰고, 불확실할
때만 비싼 경로로 넘긴다"는 고전적인 cascade 라우팅 패턴.

주의(알려진 한계):
1. 이 트리거는 router_rag/agentic_rag 각각의 실측 실패 사례([[rag_stat_query_routing]],
   [[agentic_rag_design]], [[agentic_rag_loose_case_root_causes]] 메모 참고)에서 나온 휴리스틱이지
   완벽한 신뢰도 추정이 아니다. 특히 "구조화 검색 결과 0건"은 포지션/클럽 필터가 정말로
   해당하는 선수가 없는 정상 케이스와 구분이 안 될 수 있음.
2. router_rag와 agentic_rag는 대화 history를 각자 다른 형식(OpenAI dict list vs LangChain
   message list)으로 독립 관리한다. 한 대화 안에서 어떤 turn은 router가, 다른 turn은
   agentic이 답하면 "그 중에서 ~" 같은 후속 질문이 상대방 tier의 이전 답을 못 볼 수 있다.
   지금은 에스컬레이션을 새 질문(use_previous_candidates=False)에서만 트리거해서 이 문제를
   최대한 피했지만, 완전히 막지는 못한다 (structured 결과 0건 트리거는 후속 질문에서도 발생 가능).
   완전한 해결은 두 파이프라인의 상태를 하나로 통합하는 더 큰 리팩터링이 필요 — 지금은 범위 밖.
"""

from agents.router_rag import RouterRAG
from agents.agentic_rag import AgenticRAG
from tools.retriever import FAISSRetriever

ESCALATION_REASONS = {
    "ambiguous_parse": "파서가 스탯 조건을 못 뽑아냄 (metric/filters 둘 다 비어있음)",
    "empty_structured_result": "구조화 검색 결과가 0건",
}


def needs_escalation(spec: dict, candidates: list[str]) -> tuple[bool, str | None]:
    """router_rag의 파서 출력(spec)과 구조화 검색 결과만 보고 agentic_rag 에스컬레이션
    여부를 판단하는 순수 함수 (OpenAI 호출 없음 — 유닛 테스트 가능)."""
    is_stat_query = spec.get("is_stat_query", False)
    use_previous = spec.get("use_previous_candidates", False)

    if is_stat_query and not use_previous:
        metric_empty = spec.get("metric", "NONE") == "NONE"
        filters_empty = not spec.get("filters")
        if metric_empty and filters_empty:
            return True, "ambiguous_parse"

    if is_stat_query and not candidates:
        return True, "empty_structured_result"

    return False, None


class CascadeRAG:
    def __init__(
        self,
        vector_retriever: FAISSRetriever,
        router_model: str = "gpt-4o",
        parser_model: str = "gpt-4o-mini",
        agentic_model: str = "gpt-4o-mini",
    ):
        self.router = RouterRAG(vector_retriever, model=router_model, parser_model=parser_model)
        self.agentic = AgenticRAG(vector_retriever, model=agentic_model)

    def query(self, question: str, reset_history: bool = False) -> dict:
        if reset_history:
            self.router.reset()
            self.agentic.reset()

        # 1차: router_rag의 파서만 먼저 돌려서 spec/구조화 결과를 확인한다.
        retrieval = self.router.retrieve(question, reset=False)
        escalate, reason = needs_escalation(retrieval["spec"], retrieval["candidates"])

        if escalate:
            result = self.agentic.query(question, reset_history=False)
            return {
                "question": question,
                "answer": result["answer"],
                "tier": "agentic",
                "escalation_reason": reason,
                "retrieved": result["retrieved_players"],
            }

        # 에스컬레이션 안 하면 router_rag 정식 경로(답변 생성까지)를 그대로 탄다.
        # retrieve()를 한 번 더 안에서 돌리게 되어 파서 호출이 중복되지만(gpt-4o-mini라 비용
        # 적음), router_rag.py를 건드리지 않고 조합만으로 cascade를 얹기 위한 트레이드오프다.
        result = self.router.query(question, reset_history=False)
        return {
            "question": question,
            "answer": result["answer"],
            "tier": "router",
            "escalation_reason": None,
            "retrieved": result["retrieved"],
        }

    def reset(self):
        self.router.reset()
        self.agentic.reset()


# ── 간단 테스트 ──────────────────────────────────────────
if __name__ == "__main__":
    from tools.embedder import OpenAIEmbedder

    embedder = OpenAIEmbedder()
    vec_retriever = FAISSRetriever(embedder).build()
    rag = CascadeRAG(vec_retriever)

    print("=== Cascade RAG 테스트 ===\n")
    q1 = "태클 많은 미드필더 3명 추천해줘"
    r1 = rag.query(q1)
    print(f"Q: {q1}\n[tier: {r1['tier']}]\nA: {r1['answer']}\n")

    q2 = "그 중에서 패스 성공률이 제일 높은 선수는?"
    r2 = rag.query(q2)
    print(f"Q: {q2}\n[tier: {r2['tier']}]\nA: {r2['answer']}\n")
