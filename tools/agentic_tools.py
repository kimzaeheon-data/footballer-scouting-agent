"""
Agentic RAG용 툴 정의
- stat_query   : 구조화 필터/정렬 (router_rag의 StructuredRetriever 재사용). 파서 LLM 호출 없이
                 에이전트 자신의 tool-calling이 곧 파싱이다 — router_rag보다 한 단계 더 "agentic".
- vector_search: 기존 FAISS 임베딩 검색 (서술형/애매한 질문 폴백)
- player_lookup: 선수 이름으로 직접 조회 (기존 두 파이프라인엔 없던 기능)

모든 툴은 JSON 문자열을 반환한다 — 에이전트가 읽기도 쉽고, 평가 스크립트가
ToolMessage에서 retrieved player 목록을 안정적으로 파싱하기에도 좋다.
"""

import json
import numpy as np
from typing import Literal, Optional
from pydantic import BaseModel
from langchain_core.tools import tool

from tools.retriever import FAISSRetriever
from tools.structured_query import STAT_COLUMNS, StructuredRetriever

_struct_retriever = StructuredRetriever()
_vector_retriever: Optional[FAISSRetriever] = None


def set_vector_retriever(retriever: FAISSRetriever):
    """앱 초기화 시 한 번 주입 (임베딩 모델 로딩 비용을 피하려고 전역으로 둠)."""
    global _vector_retriever
    _vector_retriever = retriever


def _json_default(o):
    """pandas/numpy 스칼라(int64, float64 등)를 json.dumps가 처리할 수 있게 변환."""
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    return str(o)


FilterOp = Literal[">=", "<=", "==", ">", "<", ">=pct", "<=pct"]
# 컬럼명을 enum으로 강제해서 존재하지 않는 컬럼을 모델이 지어내지 못하게 막는다
_ColumnEnum = Literal[tuple(STAT_COLUMNS)]
_MetricEnum = Literal[tuple(STAT_COLUMNS + ["NONE"])]


class FilterItem(BaseModel):
    column: _ColumnEnum
    op: FilterOp
    value: float


@tool
def stat_query(
    position: Literal["MID", "DEF", "FWD", "GKP", "ANY"],
    metric: _MetricEnum,
    sort_direction: Literal["desc", "asc"],
    top_n: int,
    filters: Optional[list[FilterItem]] = None,
    player_names: Optional[list[str]] = None,
) -> str:
    """프리미어리그 2024-25 선수 스탯 테이블을 직접 필터링/정렬해서 정확한 순위를 구한다.
    "가장 많은/높은 X는?", "A 이상이면서 B도 높은 선수는?" 같은 랭킹·필터 질문에 사용.

    Args:
        position: 포지션 필터 (MID/DEF/FWD/GKP), 상관없으면 ANY.
        metric: 정렬 기준 컬럼명 (사용 가능한 컬럼 목록은 tool description 참고).
            정렬이 필요 없으면(예: 이미 알고 있는 선수 목록을 그대로 유지) "NONE".
        sort_direction: metric 기준 desc(내림차순, "가장 많은/높은") 또는 asc(오름차순, "가장 적은/낮은").
        top_n: 반환할 선수 수.
        filters: [{"column": "<컬럼명>", "op": ">=|<=|==|>|<|>=pct|<=pct", "value": <숫자>}] 형태의 추가 조건.
            op이 ">=pct"/"<=pct"면 value는 0~100 백분위수를 의미 (예: 상위 50% 이상 -> value=50).
            명시적 숫자가 없는 정성적 조건("~도 높은편")은 percentile 필터(기본 50)를 사용할 것.
        player_names: 이전 turn에서 이미 좁혀놓은 후보 선수 이름 리스트. 대화의 후속 질문
            ("그 중에서 ~한 선수는?")에서 이 값을 넘기면 position/이 리스트로 한정해서
            그 안에서만 필터/정렬한다 (이 경우 position은 무시됨).
    """
    spec = {
        "position": position,
        "metric": metric,
        "sort_direction": sort_direction,
        "filters": [f.model_dump() for f in filters] if filters else [],
        "top_n": top_n,
        "use_previous_candidates": bool(player_names),
    }
    results, candidates = _struct_retriever.execute(spec, previous_candidates=player_names)
    return json.dumps(
        {
            "players": candidates,
            "detail": [r["metadata"] for r in results],
        },
        ensure_ascii=False,
        default=_json_default,
    )


@tool
def vector_search(query: str, top_k: int = 5) -> str:
    """스탯 필터/정렬로 표현하기 애매한 서술형·의미 기반 질문에 대해 임베딩 유사도 검색을 수행한다.
    예: "리버풀 스타일에 맞는 미드필더", "수비형이지만 공격 가담도 하는 선수" 처럼 정형화된
    컬럼 조건으로 못 바꾸는 질문에만 사용. 스탯 순위/필터 질문에는 stat_query를 써라.

    Args:
        query: 검색할 자연어 질의.
        top_k: 반환할 결과 수.
    """
    if _vector_retriever is None:
        return json.dumps({"error": "vector retriever not initialized"}, ensure_ascii=False)
    results = _vector_retriever.search(query, top_k=top_k)
    return json.dumps(
        {
            "players": [r["metadata"]["player"] for r in results],
            "detail": [r["metadata"] for r in results],
        },
        ensure_ascii=False,
        default=_json_default,
    )


@tool
def player_lookup(player_name: str) -> str:
    """특정 선수 이름으로 전체 스탯을 직접 조회한다. 사용자가 특정 선수를 이름으로 콕 집어
    물어볼 때 (랭킹/추천이 아니라) 사용. 데이터에 없는 선수면 없다고 알려준다.

    Args:
        player_name: 조회할 선수 이름 (한글/영문 모두 시도해볼 것).
    """
    df = _struct_retriever.df
    mask = df["Player Name"].str.contains(player_name, case=False, na=False, regex=False)
    matches = df[mask]
    if matches.empty:
        return json.dumps({"players": [], "detail": [], "found": False}, ensure_ascii=False)

    row = matches.iloc[0]
    detail = {c: row[c] for c in row.index if not c.endswith("_p90")}
    return json.dumps(
        {"players": [row["Player Name"]], "detail": [detail], "found": True},
        ensure_ascii=False,
        default=_json_default,
    )


# @tool 데코레이터는 함수의 static docstring만 읽으므로, STAT_COLUMNS처럼 동적인 목록은
# 데코레이션 이후에 description에 덧붙인다 (f-string은 파이썬 docstring으로 인식되지 않음).
stat_query.description += "\n\n사용 가능한 metric/column 목록: " + ", ".join(STAT_COLUMNS)

ALL_TOOLS = [stat_query, vector_search, player_lookup]
