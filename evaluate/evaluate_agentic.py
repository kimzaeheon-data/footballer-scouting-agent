"""
Agentic RAG 평가 스크립트 (naive/router와 동일한 지표로 3자 비교)
메인 지표 : Hit Rate @5
보조 지표 : MRR, Context Precision, Context Recall

router_rag와 달리 파싱 전용 LLM 호출이 따로 없고, 매 질문마다 ReAct 루프(추론→툴 호출→
최종 답변) 전체가 돈다. 즉 이 스크립트는 retrieval만 평가하는 게 아니라 "에이전트가 실제로
올바른 툴을 올바른 인자로 호출했는가"까지 함께 검증한다 — 비용/속도가 router_rag 평가보다 크다.
"""

import json
import os
from dataclasses import dataclass, field
from tools.retriever import FAISSRetriever
from tools.embedder import OpenAIEmbedder
from agents.agentic_rag import AgenticRAG

GOLDEN_PATH = "data/golden_set.json"
TOP_K = 5


@dataclass
class EvalResult:
    name: str
    total: int = 0
    hits: int = 0
    rr_sum: float = 0.0
    precision_sum: float = 0.0
    recall_sum: float = 0.0
    skipped: int = 0
    tool_counts: dict = field(default_factory=dict)
    type_hits: dict = field(default_factory=lambda: {"single": [], "complex": [], "followup": []})
    miss_log: list = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    @property
    def mrr(self) -> float:
        return self.rr_sum / self.total if self.total else 0.0

    @property
    def context_precision(self) -> float:
        return self.precision_sum / self.total if self.total else 0.0

    @property
    def context_recall(self) -> float:
        return self.recall_sum / self.total if self.total else 0.0


def compute_metrics(retrieved_players: list[str], expected_players: list[str], k: int = TOP_K):
    top_k = retrieved_players[:k]
    hit = int(any(p in top_k for p in expected_players))

    rr = 0.0
    for i, p in enumerate(retrieved_players):
        if p in expected_players:
            rr = 1 / (i + 1)
            break

    relevant = sum(1 for p in retrieved_players if p in expected_players)
    precision = relevant / len(retrieved_players) if retrieved_players else 0.0
    recall = relevant / len(expected_players) if expected_players else 0.0

    return hit, rr, precision, recall


def _log_miss(item_id: str, question: str, expected: list, retrieved: list, all_tool_players: list, tools_used: list, answer: str):
    print(f"\n--- MISS [{item_id}] {question}")
    print(f"    expected         : {expected}")
    print(f"    retrieved(last)  : {retrieved}")
    print(f"    all_tool_players : {all_tool_players}")
    print(f"    tools            : {tools_used}")
    print(f"    answer           : {answer[:200]}")


def _log_loose(item_id: str, question: str, expected: list, retrieved: list, precision: float, recall: float):
    # hit=1 (Hit Rate엔 안 잡힘)이지만 precision/recall이 낮은 경우 — MRR/Precision/Recall을
    # 갉아먹는 케이스라 Hit Rate만 볼 땐 안 보이던 문제를 여기서 잡는다.
    print(f"\n~~~ LOOSE [{item_id}] {question}  (hit=1, precision={precision:.2f}, recall={recall:.2f})")
    print(f"    expected         : {expected}")
    print(f"    retrieved(last)  : {retrieved}")


def evaluate(rag: AgenticRAG, golden_data: list[dict], verbose: bool = True) -> EvalResult:
    result = EvalResult(name="agentic-rag")

    for item in golden_data:
        q_type = item["type"]

        if q_type == "refusal":
            result.skipped += 1
            continue

        if q_type == "followup":
            rag.reset()
            last_retrieved: list[str] = []  # 직전에 tool이 실제로 채운 retrieved_players
            for i, turn in enumerate(item["conversation"]):
                question = turn["question"]
                expected = turn["expected_players"]
                if not expected:
                    continue

                r = rag.query(question, reset_history=False)
                for t in r["tools_used"]:
                    result.tool_counts[t] = result.tool_counts.get(t, 0) + 1

                # 이번 turn에 tool을 안 불렀으면(대화 기록만으로 답함) 직전 tool 호출 결과를
                # 채점 근거로 그대로 이어받는다 — 실제로 에이전트가 그 근거로 답한 게 맞기 때문.
                scored_players = r["retrieved_players"] if r["retrieved_players"] else last_retrieved
                if r["retrieved_players"]:
                    last_retrieved = r["retrieved_players"]

                hit, rr, precision, recall = compute_metrics(scored_players, expected)
                if verbose and not hit:
                    _log_miss(item["id"], question, expected, scored_players, r["all_tool_players"], r["tools_used"], r["answer"])
                elif verbose and hit and (precision < 0.6 or recall < 1.0):
                    _log_loose(item["id"], question, expected, scored_players, precision, recall)
                if not hit:
                    result.miss_log.append({
                        "id": item["id"], "type": "followup", "question": question,
                        "expected": expected, "retrieved": scored_players,
                        "all_tool_players": r["all_tool_players"], "tools_used": r["tools_used"],
                        "answer": r["answer"], "precision": precision, "recall": recall,
                    })
                result.total += 1
                result.hits += hit
                result.rr_sum += rr
                result.precision_sum += precision
                result.recall_sum += recall
                result.type_hits["followup"].append(hit)

        else:
            rag.reset()
            question = item["question"]
            expected = item["expected_players"]
            if not expected:
                continue

            r = rag.query(question, reset_history=True)
            for t in r["tools_used"]:
                result.tool_counts[t] = result.tool_counts.get(t, 0) + 1

            hit, rr, precision, recall = compute_metrics(r["retrieved_players"], expected)
            if verbose and not hit:
                _log_miss(item["id"], question, expected, r["retrieved_players"], r["all_tool_players"], r["tools_used"], r["answer"])
            elif verbose and hit and (precision < 0.6 or recall < 1.0):
                _log_loose(item["id"], question, expected, r["retrieved_players"], precision, recall)
            if not hit:
                result.miss_log.append({
                    "id": item["id"], "type": q_type, "question": question,
                    "expected": expected, "retrieved": r["retrieved_players"],
                    "all_tool_players": r["all_tool_players"], "tools_used": r["tools_used"],
                    "answer": r["answer"], "precision": precision, "recall": recall,
                })
            result.total += 1
            result.hits += hit
            result.rr_sum += rr
            result.precision_sum += precision
            result.recall_sum += recall
            result.type_hits[q_type].append(hit)

    return result


def print_report(router_json: dict, result: EvalResult):
    print("\n" + "=" * 70)
    print("Router RAG vs Agentic RAG 비교")
    print("=" * 70)

    header = f"{'지표':<25}{'router-rag':>20}{'agentic-rag':>20}"
    print(header)
    print("-" * 70)

    rows = [
        ("Hit Rate @5 (main)", router_json.get("hit_rate_at_5"), result.hit_rate),
        ("MRR", router_json.get("mrr"), result.mrr),
        ("Context Precision", router_json.get("context_precision"), result.context_precision),
        ("Context Recall", router_json.get("context_recall"), result.context_recall),
    ]
    for label, router_val, agentic_val in rows:
        router_str = f"{router_val:.4f}" if router_val is not None else "N/A"
        print(f"{label:<25}{router_str:>20}{agentic_val:>20.4f}")

    print("\n── 유형별 Hit Rate (agentic-rag) ──")
    router_types = router_json.get("type_hit_rate", {})
    for q_type in ["single", "complex", "followup"]:
        hits = result.type_hits[q_type]
        val = f"{sum(hits)/len(hits):.4f} ({sum(hits)}/{len(hits)})" if hits else "N/A"
        router_val = router_types.get(q_type)
        router_str = f"{router_val:.4f}" if router_val is not None else "N/A"
        print(f"{q_type:<25}{router_str:>20}{val:>20}")

    print(f"\n툴 사용 횟수: {result.tool_counts}")
    print("=" * 70)

    output = {
        "hit_rate_at_5": round(result.hit_rate, 4),
        "mrr": round(result.mrr, 4),
        "context_precision": round(result.context_precision, 4),
        "context_recall": round(result.context_recall, 4),
        "total_questions": result.total,
        "type_hit_rate": {
            t: round(sum(v) / len(v), 4) if v else None
            for t, v in result.type_hits.items()
        },
        "tool_counts": result.tool_counts,
    }
    os.makedirs("output", exist_ok=True)
    with open("output/eval_agentic_rag.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("\n결과 저장: output/eval_agentic_rag.json")

    with open("output/agentic_miss_log.json", "w", encoding="utf-8") as f:
        json.dump(result.miss_log, f, ensure_ascii=False, indent=2)
    print(f"미스 상세 로그 저장: output/agentic_miss_log.json ({len(result.miss_log)}건)")


if __name__ == "__main__":
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)["data"]

    embedder = OpenAIEmbedder()
    vec_retriever = FAISSRetriever(embedder).build()
    rag = AgenticRAG(vec_retriever, model="gpt-4o-mini")

    print("[agentic-rag] 평가 시작... (질문마다 ReAct 루프 전체가 돌아서 router-rag보다 느립니다)")
    result = evaluate(rag, golden)

    router_json_path = "output/eval_router_rag.json"
    router_json = {}
    if os.path.exists(router_json_path):
        with open(router_json_path, "r", encoding="utf-8") as f:
            router_json = json.load(f)

    print_report(router_json, result)
