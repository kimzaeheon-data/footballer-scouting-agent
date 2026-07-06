"""
Router RAG 전체 파이프라인(검색+생성) 평가 스크립트.

evaluate_router.py는 rag.retrieve()만 태워서 검색 단계만 본다 (비용/속도 때문에 의도적으로
생성 단계를 스킵). 하지만 실제 사용자는 최종 GPT-4o 답변 텍스트를 보므로, 검색이 맞아도
생성 단계에서 엉뚱한 선수를 언급하거나 빠뜨리면 실사용 품질은 떨어진다. 이 스크립트는
evaluate_refusal.py처럼 rag.query() 전체 파이프라인을 태워서 세 가지를 함께 본다:

1. retrieval_hit  : evaluate_router.py와 동일한 방식(검색 결과 top-5 안에 정답 선수 존재 여부)
2. answer_hit     : 최종 생성된 답변 텍스트 안에 정답 선수 이름이 실제로 언급됐는지
                     (검색은 맞았는데 생성 단계에서 놓치거나 다른 이름을 댄 케이스를 잡아냄)
3. faithfulness   : 답변에 등장하는 숫자들이 실제 retrieved 데이터에 있는 값인지 (숫자 hallucination
                     탐지). BLEU/ROUGE 같은 n-gram 지표는 골든 정답 "문장"이 없고(golden_set.json엔
                     선수 이름만 있음) 표현이 달라도 되는 사실 기반 태스크라 안 맞아서 대신 이 방식을
                     쓴다. 정규식 기반 휴리스틱이라 100% 정확하진 않음 — evaluate_refusal.py와 같은
                     철학으로 "자동 채점"이 아니라 "사람이 훑어볼 후보 목록"으로 취급할 것.

refusal 10문항은 evaluate_refusal.py에서 이미 별도로 검증했으므로 여기서는 skip한다.
single/complex/followup 69문항 대상. LLM 호출이 많아 evaluate_router.py보다 훨씬 느리고 비용도 크다.
"""

import json
import os
import re
from dataclasses import dataclass, field
from tools.retriever import FAISSRetriever
from tools.embedder import OpenAIEmbedder
from agents.router_rag import RouterRAG

GOLDEN_PATH = "data/golden_set.json"
TOP_K = 5
_META_TEXT_KEYS = {"player", "club", "position", "nationality"}


@dataclass
class EvalResult:
    total: int = 0
    retrieval_hits: int = 0
    answer_hits: int = 0
    rr_sum: float = 0.0
    precision_sum: float = 0.0
    recall_sum: float = 0.0
    mode_counts: dict = field(default_factory=lambda: {"structured": 0, "vector": 0})
    type_retrieval_hits: dict = field(default_factory=lambda: {"single": [], "complex": [], "followup": []})
    type_answer_hits: dict = field(default_factory=lambda: {"single": [], "complex": [], "followup": []})
    miss_log: list = field(default_factory=list)  # retrieval 자체가 틀린 케이스
    mismatch_log: list = field(default_factory=list)  # retrieval은 맞았는데 답변 텍스트가 놓친 케이스
    hallucination_log: list = field(default_factory=list)  # 답변에 근거 없는 숫자가 있는 케이스
    faithfulness_rate_sum: float = 0.0
    faithfulness_checked: int = 0

    @property
    def retrieval_hit_rate(self) -> float:
        return self.retrieval_hits / self.total if self.total else 0.0

    @property
    def answer_hit_rate(self) -> float:
        return self.answer_hits / self.total if self.total else 0.0

    @property
    def mrr(self) -> float:
        return self.rr_sum / self.total if self.total else 0.0

    @property
    def context_precision(self) -> float:
        return self.precision_sum / self.total if self.total else 0.0

    @property
    def context_recall(self) -> float:
        return self.recall_sum / self.total if self.total else 0.0

    @property
    def faithfulness_rate(self) -> float:
        return self.faithfulness_rate_sum / self.faithfulness_checked if self.faithfulness_checked else None


def compute_retrieval_metrics(retrieved_players: list[str], expected_players: list[str], k: int = TOP_K):
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


def compute_answer_hit(answer: str, expected_players: list[str]) -> bool:
    """정답 선수 이름 중 최소 1명이라도 답변 텍스트에 실제로 등장하는지 (대소문자 무시)."""
    answer_lower = (answer or "").lower()
    return any(p.lower() in answer_lower for p in expected_players)


def _num_variants(value) -> set[str]:
    """숫자 하나를 여러 표기(정수/소수1자리/원본)로 정규화 — LLM이 반올림해서 말해도 매칭되게."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return set()
    variants = {str(value)}
    if f == int(f):
        variants.add(str(int(f)))
    else:
        variants.add(f"{f:.1f}")
        variants.add(str(round(f)))
    variants.add(f"{f:.2f}".rstrip("0").rstrip("."))
    return variants


def extract_context_numbers(retrieved: list[dict]) -> set[str]:
    """retrieved 결과의 metadata에 있는 숫자값 전부 + 같은 선수 내 숫자 두 개의 합(합산 질문 대비)."""
    known: set[str] = set()
    for r in retrieved:
        meta = r.get("metadata", {})
        player_nums = []
        for k, v in meta.items():
            if k in _META_TEXT_KEYS:
                continue
            variants = _num_variants(v)
            if variants:
                known |= variants
                try:
                    player_nums.append(float(v))
                except (TypeError, ValueError):
                    pass
        # "골+어시스트 합산은?" 같은 파생 질문 대비: 같은 선수 숫자 필드 2개 조합의 합도 허용
        for i in range(len(player_nums)):
            for j in range(i + 1, len(player_nums)):
                known |= _num_variants(player_nums[i] + player_nums[j])
    return known


_LIST_MARKER_RE = re.compile(r"^\s*\d+\.\s*", re.MULTILINE)
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def extract_answer_numbers(answer: str) -> list[str]:
    """답변 텍스트에서 숫자만 뽑는다. 목록 번호("1. **선수**")는 사실이 아니므로 먼저 제거."""
    if not answer:
        return []
    cleaned = _LIST_MARKER_RE.sub("", answer)
    return _NUMBER_RE.findall(cleaned)


def check_faithfulness(answer: str, retrieved: list[dict]) -> dict:
    known = extract_context_numbers(retrieved)
    answer_numbers = extract_answer_numbers(answer)
    # 한 자리 숫자는 순위/횟수 표현("1위", "3명")일 때가 많아 노이즈가 커서 검증 대상에서 제외
    checked = [n for n in answer_numbers if len(n.replace(".", "")) >= 2]
    unverified = [n for n in checked if n not in known]
    return {
        "answer_numbers": answer_numbers,
        "checked_numbers": checked,
        "unverified_numbers": unverified,
        "faithfulness_rate": (len(checked) - len(unverified)) / len(checked) if checked else None,
    }


def _record(result: EvalResult, item_id: str, q_type: str, question: str, expected: list,
            retrieved_players: list, retrieved_full: list, answer: str, mode: str):
    retrieval_hit, rr, precision, recall = compute_retrieval_metrics(retrieved_players, expected)
    answer_hit = compute_answer_hit(answer, expected)
    faith = check_faithfulness(answer, retrieved_full)

    result.total += 1
    result.retrieval_hits += retrieval_hit
    result.answer_hits += int(answer_hit)
    result.rr_sum += rr
    result.precision_sum += precision
    result.recall_sum += recall
    result.mode_counts[mode] += 1
    result.type_retrieval_hits[q_type if q_type != "followup" else "followup"].append(retrieval_hit)
    result.type_answer_hits[q_type if q_type != "followup" else "followup"].append(int(answer_hit))
    if faith["faithfulness_rate"] is not None:
        result.faithfulness_rate_sum += faith["faithfulness_rate"]
        result.faithfulness_checked += 1

    record = {
        "id": item_id, "type": q_type, "question": question, "expected": expected,
        "retrieved": retrieved_players, "mode": mode, "answer": answer,
        "retrieval_hit": bool(retrieval_hit), "answer_hit": answer_hit,
        "precision": precision, "recall": recall,
        "faithfulness": faith,
    }

    if not retrieval_hit:
        print(f"\n--- RETRIEVAL MISS [{item_id}] {question}")
        print(f"    expected: {expected}  got: {retrieved_players}  mode: {mode}")
        result.miss_log.append(record)
    elif not answer_hit:
        # 검색은 맞았는데 최종 답변 텍스트에 정답 이름이 안 보이는 케이스 — 생성 단계 문제
        print(f"\n~~~ ANSWER MISMATCH [{item_id}] {question} (retrieval ok, answer text 못 찾음)")
        print(f"    expected: {expected}  retrieved: {retrieved_players}")
        print(f"    answer(앞부분): {(answer or '')[:200]}")
        result.mismatch_log.append(record)

    if faith["unverified_numbers"]:
        print(f"\n??? POSSIBLE HALLUCINATION [{item_id}] {question}")
        print(f"    unverified numbers: {faith['unverified_numbers']}")
        result.hallucination_log.append(record)


def evaluate(rag: RouterRAG, golden_data: list[dict]) -> EvalResult:
    result = EvalResult()

    for item in golden_data:
        q_type = item["type"]
        if q_type == "refusal":
            continue  # evaluate_refusal.py에서 별도 검증

        if q_type == "followup":
            rag.reset()
            for i, turn in enumerate(item["conversation"]):
                question = turn["question"]
                expected = turn["expected_players"]
                if not expected:
                    continue
                r = rag.query(question, reset_history=(i == 0))
                retrieved_players = [x["metadata"]["player"] for x in r["retrieved"]]
                _record(result, item["id"], "followup", question, expected,
                        retrieved_players, r["retrieved"], r["answer"], r["retrieval_mode"])
        else:
            rag.reset()
            question = item["question"]
            expected = item["expected_players"]
            if not expected:
                continue
            r = rag.query(question, reset_history=True)
            retrieved_players = [x["metadata"]["player"] for x in r["retrieved"]]
            _record(result, item["id"], q_type, question, expected,
                    retrieved_players, r["retrieved"], r["answer"], r["retrieval_mode"])

    return result


def print_report(result: EvalResult):
    print("\n" + "=" * 70)
    print("Router RAG 전체 파이프라인(검색+생성) 평가 결과")
    print("=" * 70)

    print(f"{'지표':<25}{'값':>15}")
    print("-" * 40)
    print(f"{'Retrieval Hit Rate @5':<25}{result.retrieval_hit_rate:>15.4f}")
    print(f"{'Answer Hit Rate':<25}{result.answer_hit_rate:>15.4f}")
    print(f"{'MRR (retrieval)':<25}{result.mrr:>15.4f}")
    print(f"{'Context Precision':<25}{result.context_precision:>15.4f}")
    print(f"{'Context Recall':<25}{result.context_recall:>15.4f}")
    fr = result.faithfulness_rate
    fr_str = f"{fr:.4f}" if fr is not None else "N/A"
    print(f"{'Faithfulness (숫자 근거성)':<25}{fr_str:>15}  (숫자 언급된 {result.faithfulness_checked}건 기준)")

    print("\n── 유형별 Retrieval / Answer Hit Rate ──")
    for q_type in ["single", "complex", "followup"]:
        rh = result.type_retrieval_hits[q_type]
        ah = result.type_answer_hits[q_type]
        r_str = f"{sum(rh)/len(rh):.4f} ({sum(rh)}/{len(rh)})" if rh else "N/A"
        a_str = f"{sum(ah)/len(ah):.4f} ({sum(ah)}/{len(ah)})" if ah else "N/A"
        print(f"{q_type:<12} retrieval={r_str:<18} answer={a_str}")

    print(f"\n검색 모드 사용 횟수: {result.mode_counts}")
    print(f"Retrieval MISS: {len(result.miss_log)}건 / Answer MISMATCH(검색O 답변X): {len(result.mismatch_log)}건 "
          f"/ 숫자 근거 미확인(review 필요): {len(result.hallucination_log)}건")
    print("=" * 70)

    output = {
        "retrieval_hit_rate_at_5": round(result.retrieval_hit_rate, 4),
        "answer_hit_rate": round(result.answer_hit_rate, 4),
        "mrr": round(result.mrr, 4),
        "context_precision": round(result.context_precision, 4),
        "context_recall": round(result.context_recall, 4),
        "faithfulness_rate": round(fr, 4) if fr is not None else None,
        "faithfulness_checked_count": result.faithfulness_checked,
        "total_questions": result.total,
        "type_retrieval_hit_rate": {
            t: round(sum(v) / len(v), 4) if v else None
            for t, v in result.type_retrieval_hits.items()
        },
        "type_answer_hit_rate": {
            t: round(sum(v) / len(v), 4) if v else None
            for t, v in result.type_answer_hits.items()
        },
        "mode_counts": result.mode_counts,
    }
    os.makedirs("output", exist_ok=True)
    with open("output/eval_router_rag_full.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("\n결과 저장: output/eval_router_rag_full.json")

    with open("output/router_full_miss_log.json", "w", encoding="utf-8") as f:
        json.dump({
            "retrieval_miss": result.miss_log,
            "answer_mismatch": result.mismatch_log,
            "possible_hallucination": result.hallucination_log,
        }, f, ensure_ascii=False, indent=2)
    print(f"상세 로그 저장: output/router_full_miss_log.json "
          f"(retrieval_miss {len(result.miss_log)}건, answer_mismatch {len(result.mismatch_log)}건, "
          f"possible_hallucination {len(result.hallucination_log)}건)")


if __name__ == "__main__":
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)["data"]

    embedder = OpenAIEmbedder()
    vec_retriever = FAISSRetriever(embedder).build()
    rag = RouterRAG(vec_retriever)

    print("[router-rag-full] 평가 시작... (질문마다 파싱+검색+GPT-4o 생성까지 전부 돌아서 느립니다)")
    result = evaluate(rag, golden)
    print_report(result)
