"""
Naive RAG 평가 스크립트
메인 지표 : Hit Rate @5
보조 지표 : MRR, Context Precision, Context Recall
비교 대상 : BGE-M3 vs OpenAI text-embedding-3-small
"""

import json
import os
from dataclasses import dataclass, field
from tools.retriever import FAISSRetriever
from tools.embedder import BGEEmbedder, OpenAIEmbedder

GOLDEN_PATH = "data/golden_set.json"
TOP_K = 5


# ── 지표 계산 ─────────────────────────────────────────────
@dataclass
class EvalResult:
    embedder_name: str
    total: int = 0
    hits: int = 0
    rr_sum: float = 0.0
    precision_sum: float = 0.0
    recall_sum: float = 0.0
    skipped: int = 0          # refusal 제외 카운트
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

    # Hit Rate @K
    hit = int(any(p in top_k for p in expected_players))

    # MRR
    rr = 0.0
    for i, p in enumerate(retrieved_players):
        if p in expected_players:
            rr = 1 / (i + 1)
            break

    # Context Precision (retrieved 중 relevant 비율)
    relevant = sum(1 for p in retrieved_players if p in expected_players)
    precision = relevant / len(retrieved_players) if retrieved_players else 0.0

    # Context Recall (expected 중 retrieved에 있는 비율)
    recall = relevant / len(expected_players) if expected_players else 0.0

    return hit, rr, precision, recall


# ── 평가 실행 ─────────────────────────────────────────────
def evaluate(retriever: FAISSRetriever, golden_data: list[dict]) -> EvalResult:
    result = EvalResult(embedder_name=retriever.embedder.name)

    for item in golden_data:
        q_type = item["type"]

        # refusal은 retrieval 평가 제외
        if q_type == "refusal":
            result.skipped += 1
            continue

        # followup: 각 turn 평가
        if q_type == "followup":
            for turn in item["conversation"]:
                question = turn["question"]
                expected = turn["expected_players"]
                if not expected:
                    continue

                retrieved = retriever.search(question, top_k=TOP_K)
                retrieved_players = [r["metadata"]["player"] for r in retrieved]

                hit, rr, precision, recall = compute_metrics(retrieved_players, expected)
                result.total += 1
                result.hits += hit
                result.rr_sum += rr
                result.precision_sum += precision
                result.recall_sum += recall
                result.type_hits["followup"].append(hit)

        # single / complex
        else:
            question = item["question"]
            expected = item["expected_players"]
            if not expected:
                continue

            retrieved = retriever.search(question, top_k=TOP_K)
            retrieved_players = [r["metadata"]["player"] for r in retrieved]

            hit, rr, precision, recall = compute_metrics(retrieved_players, expected)
            result.total += 1
            result.hits += hit
            result.rr_sum += rr
            result.precision_sum += precision
            result.recall_sum += recall
            result.type_hits[q_type].append(hit)

    return result


# ── 결과 출력 ─────────────────────────────────────────────
def print_report(results: list[EvalResult]):
    print("\n" + "=" * 60)
    print("Naive RAG 평가 결과")
    print("=" * 60)

    header = f"{'지표':<25}"
    for r in results:
        header += f"{r.embedder_name:>20}"
    print(header)
    print("-" * 60)

    metrics = [
        ("Hit Rate @5 (main)", lambda r: f"{r.hit_rate:.4f}"),
        ("MRR",                lambda r: f"{r.mrr:.4f}"),
        ("Context Precision",  lambda r: f"{r.context_precision:.4f}"),
        ("Context Recall",     lambda r: f"{r.context_recall:.4f}"),
        ("총 평가 질문",        lambda r: f"{r.total}"),
        ("Refusal 제외",       lambda r: f"{r.skipped}"),
    ]

    for label, fn in metrics:
        row = f"{label:<25}"
        for r in results:
            row += f"{fn(r):>20}"
        print(row)

    print("\n── 유형별 Hit Rate ──")
    for q_type in ["single", "complex", "followup"]:
        row = f"{q_type:<25}"
        for r in results:
            hits = r.type_hits[q_type]
            val = f"{sum(hits)/len(hits):.4f} ({sum(hits)}/{len(hits)})" if hits else "N/A"
            row += f"{val:>20}"
        print(row)

    print("=" * 60)

    # JSON 저장
    output = {}
    for r in results:
        output[r.embedder_name] = {
            "hit_rate_at_5": round(r.hit_rate, 4),
            "mrr": round(r.mrr, 4),
            "context_precision": round(r.context_precision, 4),
            "context_recall": round(r.context_recall, 4),
            "total_questions": r.total,
            "type_hit_rate": {
                t: round(sum(v)/len(v), 4) if v else None
                for t, v in r.type_hits.items()
            }
        }

    os.makedirs("output", exist_ok=True)
    with open("output/eval_naive_rag.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("\n결과 저장: output/eval_naive_rag.json")


# ── 메인 ─────────────────────────────────────────────────
if __name__ == "__main__":
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)["data"]

    all_results = []

    for EmbedderClass in [BGEEmbedder, OpenAIEmbedder]:
        embedder = EmbedderClass()
        retriever = FAISSRetriever(embedder).build()
        print(f"\n[{embedder.name}] 평가 시작...")
        result = evaluate(retriever, golden)
        all_results.append(result)

    print_report(all_results)
