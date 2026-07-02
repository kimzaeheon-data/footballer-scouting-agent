"""
Router RAG 평가 스크립트 (evaluate_rag.py와 동일한 지표로 naive RAG와 직접 비교)
메인 지표 : Hit Rate @5
보조 지표 : MRR, Context Precision, Context Recall
비교 대상 : Naive RAG(벡터 검색만) vs Router RAG(구조화 라우팅 + 벡터 폴백)
"""

import json
import os
from dataclasses import dataclass, field
from tools.retriever import FAISSRetriever
from tools.embedder import OpenAIEmbedder
from agents.router_rag import RouterRAG

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
    mode_counts: dict = field(default_factory=lambda: {"structured": 0, "vector": 0})
    type_hits: dict = field(default_factory=lambda: {"single": [], "complex": [], "followup": []})

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


def _log_miss(item_id: str, question: str, expected: list, retrieved: list, retrieval: dict):
    spec = retrieval["spec"]
    print(f"\n--- MISS [{item_id}] {question}")
    print(f"    expected : {expected}")
    print(f"    got      : {retrieved}")
    print(f"    mode     : {retrieval['mode']}")
    print(
        f"    spec     : position={spec.get('position')} metric={spec.get('metric')} "
        f"sort={spec.get('sort_direction')} use_prev={spec.get('use_previous_candidates')} "
        f"filters={spec.get('filters')} top_n={spec.get('top_n')}"
    )


def evaluate(rag: RouterRAG, golden_data: list[dict], verbose: bool = True) -> EvalResult:
    result = EvalResult(name="router-rag")

    for item in golden_data:
        q_type = item["type"]

        if q_type == "refusal":
            result.skipped += 1
            continue

        if q_type == "followup":
            rag.last_candidates = []  # 새 대화 시작
            for turn in item["conversation"]:
                question = turn["question"]
                expected = turn["expected_players"]
                if not expected:
                    continue

                retrieval = rag.retrieve(question)
                retrieved_players = [r["metadata"]["player"] for r in retrieval["results"]]
                result.mode_counts[retrieval["mode"]] += 1

                hit, rr, precision, recall = compute_metrics(retrieved_players, expected)
                if verbose and not hit:
                    _log_miss(item["id"], question, expected, retrieved_players, retrieval)
                result.total += 1
                result.hits += hit
                result.rr_sum += rr
                result.precision_sum += precision
                result.recall_sum += recall
                result.type_hits["followup"].append(hit)

        else:
            rag.last_candidates = []
            question = item["question"]
            expected = item["expected_players"]
            if not expected:
                continue

            retrieval = rag.retrieve(question)
            retrieved_players = [r["metadata"]["player"] for r in retrieval["results"]]
            result.mode_counts[retrieval["mode"]] += 1

            hit, rr, precision, recall = compute_metrics(retrieved_players, expected)
            if verbose and not hit:
                _log_miss(item["id"], question, expected, retrieved_players, retrieval)
            result.total += 1
            result.hits += hit
            result.rr_sum += rr
            result.precision_sum += precision
            result.recall_sum += recall
            result.type_hits[q_type].append(hit)

    return result


def print_report(naive_json: dict, router_result: EvalResult):
    print("\n" + "=" * 70)
    print("Naive RAG vs Router RAG 비교")
    print("=" * 70)

    naive_best = max(naive_json.values(), key=lambda v: v["hit_rate_at_5"])

    header = f"{'지표':<25}{'naive (best embedder)':>25}{'router-rag':>20}"
    print(header)
    print("-" * 70)

    rows = [
        ("Hit Rate @5 (main)", naive_best["hit_rate_at_5"], router_result.hit_rate),
        ("MRR", naive_best["mrr"], router_result.mrr),
        ("Context Precision", naive_best["context_precision"], router_result.context_precision),
        ("Context Recall", naive_best["context_recall"], router_result.context_recall),
    ]
    for label, naive_val, router_val in rows:
        print(f"{label:<25}{naive_val:>25.4f}{router_val:>20.4f}")

    print("\n── 유형별 Hit Rate (router-rag) ──")
    for q_type in ["single", "complex", "followup"]:
        hits = router_result.type_hits[q_type]
        naive_val = naive_best["type_hit_rate"].get(q_type)
        val = f"{sum(hits)/len(hits):.4f} ({sum(hits)}/{len(hits)})" if hits else "N/A"
        naive_str = f"{naive_val:.4f}" if naive_val is not None else "N/A"
        print(f"{q_type:<25}{naive_str:>25}{val:>20}")

    print(f"\n검색 모드 사용 횟수: {router_result.mode_counts}")
    print("=" * 70)

    output = {
        "hit_rate_at_5": round(router_result.hit_rate, 4),
        "mrr": round(router_result.mrr, 4),
        "context_precision": round(router_result.context_precision, 4),
        "context_recall": round(router_result.context_recall, 4),
        "total_questions": router_result.total,
        "type_hit_rate": {
            t: round(sum(v) / len(v), 4) if v else None
            for t, v in router_result.type_hits.items()
        },
        "mode_counts": router_result.mode_counts,
    }
    os.makedirs("output", exist_ok=True)
    with open("output/eval_router_rag.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("\n결과 저장: output/eval_router_rag.json")


if __name__ == "__main__":
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)["data"]

    embedder = OpenAIEmbedder()
    vec_retriever = FAISSRetriever(embedder).build()
    rag = RouterRAG(vec_retriever)

    print("[router-rag] 평가 시작...")
    result = evaluate(rag, golden)

    naive_json_path = "output/eval_naive_rag.json"
    naive_json = {}
    if os.path.exists(naive_json_path):
        with open(naive_json_path, "r", encoding="utf-8") as f:
            naive_json = json.load(f)

    print_report(naive_json, result)
