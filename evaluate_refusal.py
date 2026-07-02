"""
Refusal 케이스 검증 스크립트
- 대상: golden_set.json의 refusal 10문항 (다른 리그/시즌/연봉/부상 등 데이터에 없는 정보)
- retrieve()만 쓰는 evaluate_router.py와 달리, query() 전체 파이프라인(파싱→검색→GPT-4o 답변)을
  실제로 태워서 최종 답변이 진짜로 "데이터에 없다"고 거절하는지 확인한다.
- 자동 채점기가 아니라 사람이 눈으로 보라고 정리해서 출력하는 스크립트 (키워드 매치는 참고용 힌트일 뿐).
"""

import json
import os
from tools.retriever import FAISSRetriever
from tools.embedder import OpenAIEmbedder
from agents.router_rag import RouterRAG

GOLDEN_PATH = "data/golden_set.json"

REFUSAL_HINTS = ["없습니다", "없어요", "확인할 수 없", "제공된 데이터", "데이터에 없", "포함되어 있지 않"]


def looks_like_refusal(answer: str) -> bool:
    return any(h in answer for h in REFUSAL_HINTS)


def run():
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)["data"]

    refusal_items = [item for item in golden if item["type"] == "refusal"]

    embedder = OpenAIEmbedder()
    vec_retriever = FAISSRetriever(embedder).build()
    rag = RouterRAG(vec_retriever)

    print("=" * 70)
    print(f"Refusal 케이스 검증 ({len(refusal_items)}문항)")
    print("=" * 70)

    results = []
    likely_ok = 0

    for item in refusal_items:
        rag.reset()  # 매 문항 독립 대화로 처리
        result = rag.query(item["question"], reset_history=True)
        ok = looks_like_refusal(result["answer"])
        likely_ok += int(ok)

        print(f"\n[{item['id']}] {item['question']}")
        print(f"  이유(정답 근거): {item['reason']}")
        print(f"  검색 모드: {result['retrieval_mode']}")
        print(f"  답변: {result['answer']}")
        print(f"  거절처럼 보임(키워드 기반 참고용): {'YES' if ok else 'NO — 직접 확인 필요'}")

        results.append({
            "id": item["id"],
            "question": item["question"],
            "reason": item["reason"],
            "retrieval_mode": result["retrieval_mode"],
            "answer": result["answer"],
            "looks_like_refusal": ok,
        })

    print("\n" + "=" * 70)
    print(f"키워드 기반 거절 추정: {likely_ok}/{len(refusal_items)} (참고용 — 실제로는 위 답변을 직접 읽고 판단할 것)")
    print("=" * 70)

    os.makedirs("output", exist_ok=True)
    with open("output/eval_refusal.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n결과 저장: output/eval_refusal.json")


if __name__ == "__main__":
    run()
