"""
스카우팅 어시스턴트 CLI 진입점.

RAG_COMPARISON_REPORT.md 비교 결과 router_rag를 1순위로 권장하므로 이걸 기본 파이프라인
으로 쓴다. 대화형으로 질문에 답하고, 구조화 검색(필터/정렬)으로 선수가 나온 답변 뒤에는
스카우팅 리포트 이미지 생성도 제안한다 (offer_report=True는 evaluate/ 스크립트들의 채점에
영향 주지 않으려고 옵트인으로 빼둔 옵션 — 실제 대화형 진입점인 여기서만 켠다).

사용법:
    python main.py
    (종료: exit / quit / 종료)
"""
from tools.embedder import BGEEmbedder
from tools.retriever import FAISSRetriever
from agents.router_rag import RouterRAG

EXIT_COMMANDS = {"exit", "quit", "종료", "q"}


def main():
    print("선수 데이터 인덱스 로딩 중...")
    vector_retriever = FAISSRetriever(embedder=BGEEmbedder()).build()
    rag = RouterRAG(vector_retriever=vector_retriever)

    print("\n=== 프리미어리그 2024-25 스카우팅 어시스턴트 ===")
    print("선수 관련 질문을 입력하세요. (종료: exit)\n")

    awaiting_report_reply = False
    pending_player = None

    while True:
        user_input = input("질문> ").strip()
        if not user_input:
            continue
        if user_input.lower() in EXIT_COMMANDS:
            print("종료합니다.")
            break

        if awaiting_report_reply:
            awaiting_report_reply = False
            if RouterRAG.wants_report(user_input):
                path = rag.generate_report(pending_player)
                print(f"\n스카우팅 리포트 저장됨: {path}\n")
            else:
                print()
            pending_player = None
            continue

        result = rag.query(user_input, offer_report=True)
        print(f"\n{result['answer']}\n")

        if result["report_suggestion"]:
            awaiting_report_reply = True
            # report_suggestion은 retrieval_mode == "structured"일 때만 채워지고,
            # 그 경우 query() 내부에서 호출한 retrieve()가 last_candidates를 갱신해둔 상태다.
            pending_player = rag.last_candidates[0]


if __name__ == "__main__":
    main()
