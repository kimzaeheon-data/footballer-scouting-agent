"""
강건성(robustness) 테스트 — golden_set.json 문항 12개를 골라 표현만 바꾼 변형 질문으로
router_rag를 다시 태워본다.

지금까지 모든 평가(evaluate_router.py, evaluate_router_full.py 등)는 golden_set.json에 미리
정해둔 "템플릿 문장 그대로"만 테스트했다. 실제 사용자는 반말/줄임말/오타/다른 어순 등 온갖 변형으로
묻는데, 그럴 때도 같은 정답을 내는지는 한 번도 검증한 적이 없다. 이 스크립트는 data/robustness_set.json에
정의된 12개 원본 문항 × 3개 변형(casual/typo_abbrev/reordered) = 36개 변형 질문을 RouterRAG.query()에
태워서, 원본과 같은 expected_players가 나오는지 확인한다.

evaluate_router_full.py와 같은 두 지표(retrieval_hit, answer_hit)를 쓰고, 변형 유형별로 쪼개서
어떤 스타일이 시스템을 더 잘 깨뜨리는지 본다.
"""

import json
import os
from dataclasses import dataclass, field
from tools.retriever import FAISSRetriever
from tools.embedder import OpenAIEmbedder
from agents.router_rag import RouterRAG
from evaluate_router_full import compute_retrieval_metrics, compute_answer_hit, check_faithfulness

ROBUSTNESS_PATH = "data/robustness_set.json"
TOP_K = 5


@dataclass
class EvalResult:
    total: int = 0
    retrieval_hits: int = 0
    answer_hits: int = 0
    by_variant: dict = field(default_factory=lambda: {
        "casual": {"total": 0, "retrieval_hit": 0, "answer_hit": 0},
        "typo_abbrev": {"total": 0, "retrieval_hit": 0, "answer_hit": 0},
        "reordered": {"total": 0, "retrieval_hit": 0, "answer_hit": 0},
    })
    miss_log: list = field(default_factory=list)

    @property
    def retrieval_hit_rate(self):
        return self.retrieval_hits / self.total if self.total else 0.0

    @property
    def answer_hit_rate(self):
        return self.answer_hits / self.total if self.total else 0.0


def _record(result: EvalResult, original_id: str, variant_type: str, question: str,
            expected: list, r: dict):
    retrieved_players = [x["metadata"]["player"] for x in r["retrieved"]]
    retrieval_hit, rr, precision, recall = compute_retrieval_metrics(retrieved_players, expected, k=TOP_K)
    answer_hit = compute_answer_hit(r["answer"], expected)

    result.total += 1
    result.retrieval_hits += retrieval_hit
    result.answer_hits += int(answer_hit)
    vb = result.by_variant[variant_type]
    vb["total"] += 1
    vb["retrieval_hit"] += retrieval_hit
    vb["answer_hit"] += int(answer_hit)

    record = {
        "original_id": original_id, "variant_type": variant_type, "question": question,
        "expected": expected, "retrieved": retrieved_players, "answer": r["answer"],
        "retrieval_hit": bool(retrieval_hit), "answer_hit": answer_hit,
        "precision": precision, "recall": recall,
    }
    if not retrieval_hit or not answer_hit:
        tag = "RETRIEVAL MISS" if not retrieval_hit else "ANSWER MISS"
        print(f"\n--- {tag} [{original_id}/{variant_type}] {question}")
        print(f"    expected: {expected}  got: {retrieved_players}")
        print(f"    answer(앞부분): {(r['answer'] or '')[:150]}")
        result.miss_log.append(record)


def evaluate(rag: RouterRAG, robustness_data: list[dict]) -> EvalResult:
    result = EvalResult()

    for item in robustness_data:
        oid = item["original_id"]
        q_type = item["type"]

        for variant_type, variant in item["variants"].items():
            if q_type == "followup":
                rag.reset()
                expected_per_turn = item["expected_players_per_turn"]
                for i, question in enumerate(variant):
                    expected = expected_per_turn[i]
                    if not expected:
                        continue
                    r = rag.query(question, reset_history=(i == 0))
                    _record(result, oid, variant_type, question, expected, r)
            else:
                rag.reset()
                expected = item["expected_players"]
                r = rag.query(variant, reset_history=True)
                _record(result, oid, variant_type, variant, expected, r)

    return result


def print_report(result: EvalResult):
    print("\n" + "=" * 70)
    print("강건성 테스트 결과 (golden set 템플릿 밖 표현)")
    print("=" * 70)
    print(f"전체 변형 질문: {result.total}건")
    print(f"Retrieval Hit Rate: {result.retrieval_hit_rate:.4f}")
    print(f"Answer Hit Rate   : {result.answer_hit_rate:.4f}")

    print("\n── 변형 유형별 ──")
    for vt, vb in result.by_variant.items():
        rh = vb["retrieval_hit"] / vb["total"] if vb["total"] else None
        ah = vb["answer_hit"] / vb["total"] if vb["total"] else None
        rh_str = f"{rh:.4f}" if rh is not None else "N/A"
        ah_str = f"{ah:.4f}" if ah is not None else "N/A"
        print(f"{vt:<15} retrieval={rh_str} ({vb['retrieval_hit']}/{vb['total']})  "
              f"answer={ah_str} ({vb['answer_hit']}/{vb['total']})")

    print(f"\nMISS 건수: {len(result.miss_log)}건")
    print("=" * 70)

    output = {
        "total": result.total,
        "retrieval_hit_rate": round(result.retrieval_hit_rate, 4),
        "answer_hit_rate": round(result.answer_hit_rate, 4),
        "by_variant": {
            vt: {
                "total": vb["total"],
                "retrieval_hit_rate": round(vb["retrieval_hit"] / vb["total"], 4) if vb["total"] else None,
                "answer_hit_rate": round(vb["answer_hit"] / vb["total"], 4) if vb["total"] else None,
            }
            for vt, vb in result.by_variant.items()
        },
    }
    os.makedirs("output", exist_ok=True)
    with open("output/eval_robustness.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("\n결과 저장: output/eval_robustness.json")

    with open("output/robustness_miss_log.json", "w", encoding="utf-8") as f:
        json.dump(result.miss_log, f, ensure_ascii=False, indent=2)
    print(f"상세 로그 저장: output/robustness_miss_log.json ({len(result.miss_log)}건)")


if __name__ == "__main__":
    with open(ROBUSTNESS_PATH, "r", encoding="utf-8") as f:
        robustness_data = json.load(f)["data"]

    embedder = OpenAIEmbedder()
    vec_retriever = FAISSRetriever(embedder).build()
    rag = RouterRAG(vec_retriever)

    print(f"[robustness] 평가 시작... ({len(robustness_data)}개 원본 문항 x 3개 변형)")
    result = evaluate(rag, robustness_data)
    print_report(result)
