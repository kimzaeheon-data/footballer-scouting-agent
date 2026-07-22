"""
Cascade RAG 데모 UI (Streamlit)

agents/cascade_rag.py의 CascadeRAG를 채팅 형태로 시연한다. 각 답변마다 어느 tier
(router_rag 그대로 vs agentic_rag로 에스컬레이션)가 답했는지, 에스컬레이션했다면
왜 했는지를 배지로 보여준다 — cascade 라우팅이 실제로 작동하는 걸 눈으로 확인하기 위한
용도.

실행:
    streamlit run demo_cascade.py
    (.env에 OPENAI_API_KEY 필요, 첫 실행 시 BGE-m3 임베딩 모델 로딩에 시간이 좀 걸림)
"""

import streamlit as st

from agents.cascade_rag import CascadeRAG, ESCALATION_REASONS
from tools.embedder import BGEEmbedder
from tools.retriever import FAISSRetriever

st.set_page_config(page_title="Cascade RAG 스카우팅 데모", page_icon="⚽", layout="centered")

TIER_BADGE = {
    "router": "🔵 router_rag",
    "agentic": "🟠 agentic_rag (에스컬레이션됨)",
}


@st.cache_resource(show_spinner="선수 데이터 인덱스 로딩 중... (첫 실행만 느림)")
def load_rag() -> CascadeRAG:
    vector_retriever = FAISSRetriever(embedder=BGEEmbedder()).build()
    return CascadeRAG(vector_retriever=vector_retriever)


rag = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content", "tier", "reason"}]
if "tier_counts" not in st.session_state:
    st.session_state.tier_counts = {"router": 0, "agentic": 0}

st.title("⚽ 프리미어리그 스카우팅 어시스턴트")
st.caption(
    "Cascade RAG 데모 — router_rag를 기본으로 쓰고, 파서가 애매해하거나 구조화 검색이 "
    "빈손일 때만 agentic_rag로 넘어갑니다."
)

with st.sidebar:
    st.subheader("Cascade 라우팅이란")
    st.write(
        "1차로 router_rag(싸고 빠르고 결정론적)를 태우고, 다음 신호가 보일 때만 "
        "agentic_rag(느리지만 더 유연함)로 에스컬레이션합니다."
    )
    for reason, desc in ESCALATION_REASONS.items():
        st.markdown(f"- **{reason}**: {desc}")

    st.divider()
    st.subheader("이번 세션 tier 사용 현황")
    st.metric("router_rag", st.session_state.tier_counts["router"])
    st.metric("agentic_rag (에스컬레이션)", st.session_state.tier_counts["agentic"])

    st.divider()
    if st.button("대화 초기화"):
        rag.reset()
        st.session_state.messages = []
        st.session_state.tier_counts = {"router": 0, "agentic": 0}
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            badge = TIER_BADGE[msg["tier"]]
            caption = badge
            if msg.get("reason"):
                caption += f" — {ESCALATION_REASONS.get(msg['reason'], msg['reason'])}"
            st.caption(caption)

question = st.chat_input("선수 관련 질문을 입력하세요 (예: 태클 많은 미드필더 3명 추천해줘)")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            result = rag.query(question, reset_history=False)
        st.write(result["answer"])
        badge = TIER_BADGE[result["tier"]]
        caption = badge
        if result["escalation_reason"]:
            caption += f" — {ESCALATION_REASONS.get(result['escalation_reason'], result['escalation_reason'])}"
        st.caption(caption)

    st.session_state.tier_counts[result["tier"]] += 1
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "tier": result["tier"],
            "reason": result["escalation_reason"],
        }
    )
