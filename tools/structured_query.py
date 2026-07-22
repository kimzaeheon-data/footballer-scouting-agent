"""
구조화 쿼리 라우팅
- StructuredQueryParser: LLM function-calling으로 자연어 질문 → 구조화 스펙(JSON) 변환
- StructuredRetriever  : 스펙을 pandas filter/sort로 실행 → 정확한 top-N 반환

목적:
  "가장 많은/높은 X는?" 류의 argmax·필터 질문은 임베딩 유사도 검색으로 풀 수 없다
  (텍스트가 비슷한 선수를 찾을 뿐, 수치가 실제로 가장 높은 선수를 찾지 않음).
  이 모듈은 그런 질문을 pandas 연산으로 직접 answer하여 100%에 가까운 정확도를 낸다.
  질문이 구조화 질의로 해석되지 않으면 (is_stat_query=False) 상위 라우터가
  기존 벡터 검색(FAISSRetriever)으로 폴백한다.
"""

import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

PROCESSED_PATH = "data/processed/players_clean.csv"

# ── 화이트리스트: players_clean.csv의 실제 컬럼명 (LLM이 이 안에서만 고르도록 강제) ──
STAT_COLUMNS = [
    "Appearances", "Minutes", "Goals", "Assists", "Shots", "Shots On Target",
    "Conversion %", "Big Chances Missed", "Hit Woodwork", "Offsides", "Touches",
    "Passes", "Successful Passes", "Passes%", "Crosses", "Successful Crosses",
    "Crosses %", "fThird Passes", "Successful fThird Passes", "fThird Passes %",
    "Through Balls", "Carries", "Progressive Carries", "Carries Ended with Goal",
    "Carries Ended with Assist", "Carries Ended with Shot", "Carries Ended with Chance",
    "Possession Won", "Dispossessed", "Clean Sheets", "Clearances", "Interceptions",
    "Blocks", "Tackles", "Ground Duels", "gDuels Won", "gDuels %", "Aerial Duels",
    "aDuels Won", "aDuels %", "Goals Conceded", "xGoT Conceded", "Own Goals",
    "Fouls", "Yellow Cards", "Red Cards", "Saves", "Saves %", "Penalties Saved",
    "Clearances Off Line", "Punches", "High Claims", "Goals Prevented",
]

PARSER_SYSTEM_PROMPT = f"""
너는 축구 스탯 질문을 구조화된 pandas 필터/정렬 스펙으로 변환하는 파서야.

[사용 가능한 컬럼 (이 목록 안에서만 골라야 함)]
{", ".join(STAT_COLUMNS)}

[한국어 용어 힌트]
인터셉트→Interceptions, 태클/압박/수비 강도→Tackles, 클리어런스→Clearances, 패스 성공률/정확도→Passes%,
세이브율→Saves %, 크로스 성공률→Crosses %, 공중 듀얼 승률→aDuels %, 공중 듀얼 참여→Aerial Duels,
슈팅 성공률/전환율→Conversion %, 볼 점유 획득→Possession Won, 클린시트→Clean Sheets,
출전시간/많이 뜀/주전/풀타임→Minutes (숫자 없이 "주전"만 언급되면 기본값 Minutes>=1800 사용),
프로그레시브 캐리/전진/빌드업 능력→Progressive Carries,
경고(카드)→Yellow Cards, 어시스트→Assists, 골→Goals, 세이브→Saves, 슈팅→Shots, 유효슈팅→Shots On Target,
블록→Blocks, 파울→Fouls.
포지션 힌트: 공격수/스트라이커/윙어/포워드→FWD, 미드필더/미필→MID, 수비수/센터백/풀백→DEF, 골키퍼→GKP.
맨유→Manchester United, 맨시티→Manchester City, 첼시→Chelsea, 아스날→Arsenal,
리버풀→Liverpool, 토트넘→Tottenham Hotspur,뉴캐슬→Newcastle United,
브라이튼→Brighton & Hove Albion, 아스톤빌라→Aston Villa, 울버햄튼→Wolverhampton Wanderers, 노팅엄→Nottingham Forest

[규칙]
1. is_stat_query: 질문이 특정 포지션의 선수를 스탯 기준으로 찾거나 추천/순위/필터링하는 질문이면 true.
   "가장 많은/높은"처럼 단일 최상급 표현뿐 아니라 "A 이상이면서 B도 높은", "A와 B가 모두 높은" 같은
   복합조건 질문도 true다 (최상급 단어가 없어도 실질적으로 스탯 필터/정렬 질문이면 true).
   다른 리그/시즌/연봉/부상 등 데이터에 없는 정보를 묻거나 순수 잡담이면 false.
2. use_previous_candidates: 질문이 "그 중에서", "그 선수", "그들 중" 등 직전 대화의 후보 목록을
   전제로 할 때 true. 이 경우 position은 무시되고 이전 후보 집합 내에서만 필터/정렬한다.
   랭킹 질문이 아니라 소속팀/국적처럼 단순 서술 정보를 묻는 후속 질문이어도, 직전 후보를
   전제로 하면 use_previous_candidates=true, metric=NONE으로 설정한다 (이 경우도 is_stat_query는
   질문 성격에 맞게 true/false 아무거나 정확히 판단하면 되고, 라우팅에는 영향 없음).
   단 "OO 중에서"에서 OO가 "그/그들/이들" 같은 지시대명사가 아니라 구체적인 새 엔티티(클럽명, 포지션 등)라면 -예: "맨유 선수 중에서",
   "수비수 중에서" - 이건 이전 대화와 무관하게 새 모집단을 정의하는 완전히 새 질문이다. 이 경우 use_previous_candidates=False로 두고,
   언급된 조건 (club/position)으로 처음부터 새로 필터링한다. 오직 "그/그들/이들"처럼 지시대명사가 실제로 있을 때만 이전 후보 집합을 이어 받는다.
   - "맨유 선수 중에서 가장 많은 골을 기록한 선수는?" -> use_previous_candidates=False
        (새 질문 - "맨유"라는 구체적 새 조건이 있음), club="Manchester United", metric="Goals"
   - (비교) "그 중에서 맨유 선수 있어?" -> use_previous_candidates=True (지시대명사 "그"가 직전 후보를 가리킴), club="Manchester United"
3. position: MID/DEF/FWD/GKP 중 하나. 명시 없고 use_previous_candidates도 아니면 ANY.
4. club: 클럽명이 명시되면 그 클럽, 없으면 ANY
5. metric: 정렬 기준 컬럼. 질문이 특정 스탯을 요구하지 않으면(예: 소속팀/국적을 묻는 후속 질문) NONE.
6. sort_direction: "가장 많은/높은"→desc, "가장 적은/낮은"→asc. 기본 desc.
7. 복합조건 질문 처리 ("A 이상이면서 B도 많은/높은 X", "A와 B가 모두 높은 X" 등 조건이 2개 이상):
   - 명시적 숫자가 있는 조건(예: "80% 이상", "5개 이상", "3장 이하")은 그 조건을 filters에 넣는다
     (op: >=,<=,==,>,< 와 실제 숫자 value).
   - 숫자가 없는 조건 중, 문장에서 더 나중에 언급되거나 "~도 많은/높은/좋은"처럼 강조되는 조건을
     metric(정렬 기준)으로 삼는다. metric으로 쓴 컬럼은 filters에 다시 넣지 않는다.
   - "A와 B가 모두 많은/높은 X" 패턴(대등 접속, 강조 표현 없음)도 동일하게: **먼저 언급된 A는
     percentile filter, 나중에 언급된 B는 metric**으로 고정한다. 예외 없이 이 순서를 따른다.
   - 나머지 숫자 없는 조건(주로 먼저 언급된 조건, 예: "압박이 강하면서"의 "압박")은 percentile
     필터로 처리: "많은/높은/강한" 계열이면 op=">=pct", value=50 / "적은/낮은/없이" 계열이면
     op="<=pct", value=50 (명시적 숫자가 없는 한 항상 50을 기본값으로 사용, 임의의 다른 숫자를
     추측하지 마). **주의: "카드 없이", "파울 없이"처럼 "없이/없는"이 나오더라도 절대 op="=="와
     value=0으로 처리하지 마라 — 이는 "적은 편"이라는 정성적 표현이지 완전히 0개라는 뜻이 아니다.
     반드시 op="<=pct", value=50을 사용해.**
   - 조건이 1개뿐이면 그 조건을 filters에, 남은 능력 표현(예: "빌드업 능력 있는")을 metric으로.
8. top_n: use_previous_candidates=false인 새 질문(단발/복합형)은 문장이 문법적으로 단수형("~는?")
   이어도 특정 인원수가 명시되지 않는 한 기본적으로 top_n=3을 사용해 (이 데이터셋에서는 추천 질문에
   보통 3명을 기대함). "5명", "3명"처럼 숫자가 명시되면 그 숫자를 쓴다.
   use_previous_candidates=true인 후속 질문에서는 "그 선수는?"처럼 단수로 좁히면 top_n=1,
   "그 선수들은?"처럼 복수/전체를 가리키면 이전 후보 개수만큼(top_n 생략시 큰 값 사용).
9. use_previous_candidates=true일 때 metric 판단이 특히 중요하다. 두 유형을 구분해:
   - "그 중 X가 많은/높은/적은/낮은 선수는?" (그룹 내에서 특정 기준으로 재선별) → 그 컬럼을
     그대로 metric으로 쓰고 sort_direction을 맞춘다 (많은/높은→desc, 적은/낮은→asc).
     filters는 비워둔다 — "적은"을 filters의 op="<=" 같은 걸로 바꾸지 마라, 그건 metric+asc다.
   - "그 선수 ~은/는?" (단수로 이미 정해진 1명을 다시 가리키며 설명을 요구, 특히 "합산/총합"처럼
     화이트리스트에 없는 파생 지표를 묻는 경우) → metric=NONE, filters=[]로 두고 이전 후보 순서를
     그대로 유지한 채 top_n만 적용한다. 새로운 컬럼으로 재정렬하려 하지 마라.

[예시]
- "태클과 인터셉트가 모두 높은 수비수는?" → position=DEF, filters=[{{"column":"Tackles","op":">=pct","value":50}}], metric="Interceptions"
- "블록과 클리어런스 모두 많은 수비수는?" → position=DEF, filters=[{{"column":"Blocks","op":">=pct","value":50}}], metric="Clearances"
- "주전으로 뛰면서 태클도 많은 미드필더는?" → filters=[{{"column":"Minutes","op":">=","value":1800}}], metric="Tackles"
- "경고 카드 없이 태클 많은 미드필더는?" → filters=[{{"column":"Yellow Cards","op":"<=pct","value":50}}], metric="Tackles"
- "맨유 선수들 중에서 가장 많은 골을 넣은 선수는?" → club="Manchester United", metric="Goals"
- "패스 성공률이 높으면서 어시스트도 3개 이상 기록한 미드필더는?" → filters=[{{"column":"Assists", "op":">=", "value":3}}], metric="Passes%"
- (use_previous_candidates 상황) "그 중 경고 카드 적은 선수는?" → use_previous_candidates=true, metric="Yellow Cards", sort_direction="asc", filters=[], top_n=1
- (use_previous_candidates 상황) "그 선수 골+어시스트 합산은?" → use_previous_candidates=true, metric="NONE", filters=[], top_n=1 (재정렬 금지, 이전 1위 후보 유지)
""".strip()

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "stat_query_spec",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "is_stat_query": {"type": "boolean"},
                "use_previous_candidates": {"type": "boolean"},
                "position": {"type": "string", "enum": ["MID", "DEF", "FWD", "GKP", "ANY"]},
                "club": {"type": "string", "enum": ["Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton & Hove Albion", "Chelsea",
                            "Crystal Palace", "Everton", "Fulham", "Ipswich Town", "Leicester City", "Liverpool","Manchester City", "Manchester United",
                            "Newcastle United", "Nottingham Forest", "Southampton", "Tottenham Hotspur", "West Ham United", "Wolverhampton Wanderers", "ANY"]},
                "metric": {"type": "string", "enum": STAT_COLUMNS + ["NONE"]},
                "sort_direction": {"type": "string", "enum": ["desc", "asc"]},
                "filters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string", "enum": STAT_COLUMNS},
                            "op": {"type": "string", "enum": [">=", "<=", "==", ">", "<", ">=pct", "<=pct"]},
                            "value": {"type": "number"},
                        },
                        "required": ["column", "op", "value"],
                        "additionalProperties": False,
                    },
                },
                "top_n": {"type": "integer"},
            },
            "required": [
                "is_stat_query", "use_previous_candidates", "position", "club",
                "metric", "sort_direction", "filters", "top_n",
            ],
            "additionalProperties": False,
        },
    },
}

FALLBACK_SPEC = {
    "is_stat_query": False,
    "use_previous_candidates": False,
    "position": "ANY",
    "club" : "ANY",
    "metric": "NONE",
    "sort_direction": "desc",
    "filters": [],
    "top_n": 3,
}


class StructuredQueryParser:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def parse(self, question: str, history_context: str = "") -> dict:
        user_content = question
        if history_context:
            user_content = f"[이전 대화 맥락]\n{history_context}\n\n[현재 질문]\n{question}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PARSER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format=RESPONSE_SCHEMA,
                temperature=0,
            )
            spec = json.loads(response.choices[0].message.content)
            return spec
        except Exception as e:
            print(f"[StructuredQueryParser] 파싱 실패, 벡터 검색으로 폴백: {e}")
            return dict(FALLBACK_SPEC)


class StructuredRetriever:
    def __init__(self, csv_path: str = PROCESSED_PATH):
        self.df = pd.read_csv(csv_path)

    # ── 필터 적용 + 정렬 ─────────────────────────────────
    def _apply_filters_and_sort(self, subset: pd.DataFrame, spec: dict, quantile_base: pd.DataFrame) -> pd.DataFrame:
        working = subset.copy()
        for f in spec.get("filters", []):
            col, op, val = f["column"], f["op"], f["value"]
            if col not in working.columns:
                continue
            if op == ">=pct":
                thr = quantile_base[col].quantile(val / 100)
                working = working[working[col] >= thr]
            elif op == "<=pct":
                thr = quantile_base[col].quantile(val / 100)
                working = working[working[col] <= thr]
            elif op == ">=":
                working = working[working[col] >= val]
            elif op == "<=":
                working = working[working[col] <= val]
            elif op == "==":
                working = working[working[col] == val]
            elif op == ">":
                working = working[working[col] > val]
            elif op == "<":
                working = working[working[col] < val]

        if working.empty:  # 조건이 너무 빡빡하면 완화
            working = subset.copy()

        metric = spec.get("metric")
        top_n = spec.get("top_n") or 3
        if metric and metric != "NONE" and metric in working.columns:
            ascending = spec.get("sort_direction") == "asc"
            working = working.sort_values(metric, ascending=ascending)

        return working.head(top_n)

    # ── 결과 포맷 (텍스트 + 메타데이터) ───────────────────
    def _format_row(self, row: pd.Series, spec: dict) -> dict:
        key_cols = []
        metric = spec.get("metric")
        if metric and metric != "NONE":
            key_cols.append(metric)
        for f in spec.get("filters", []):
            if f["column"] not in key_cols:
                key_cols.append(f["column"])
        for c in ["Minutes", "Goals", "Assists"]:
            if c not in key_cols:
                key_cols.append(c)

        stats_str = ", ".join(f"{c}: {row[c]}" for c in key_cols if c in row.index)
        text = f"{row['Player Name']} ({row['Club']}, {row['Position']}) - {stats_str}"
        metadata = {
            "player": row["Player Name"],
            "club": row["Club"],
            "position": row["Position"],
            "nationality": row.get("Nationality", "Unknown"),
        }
        for c in key_cols:
            if c in row.index:
                metadata[c] = row[c]
        return {"text": text, "metadata": metadata, "score": 1.0}

    # ── 실행 ─────────────────────────────────────────────
    def execute(self, spec: dict, previous_candidates: list[str] | None = None):
        """
        Returns:
            (results, candidate_names)
            results: list of {"text", "metadata", "score"} (정렬 순서 유지)
            candidate_names: 다음 turn에 넘겨줄 선수명 리스트
        """
        df = self.df

        if spec.get("use_previous_candidates") and previous_candidates:
            base = df[df["Player Name"].isin(previous_candidates)].copy()
            if base.empty:
                base = df.copy()

            club = spec.get("club", "ANY")
            if club not in (None, "ANY"):
                base = base[base["Club"] == club]

            metric = spec.get("metric")
            if not metric or metric == "NONE":
                # 정렬 기준 없음 → 이전 후보 순서 그대로 유지
                order = {name: i for i, name in enumerate(previous_candidates)}
                base["_order"] = base["Player Name"].map(order).fillna(len(previous_candidates))
                top_n = spec.get("top_n") or len(previous_candidates)
                result_df = base.sort_values("_order").head(top_n)
            else:
                result_df = self._apply_filters_and_sort(base, spec, quantile_base=base)
        else:
            position = spec.get("position", "ANY")
            base = df if position in (None, "ANY") else df[df["Position"] == position]

            club = spec.get("club", "ANY")
            if club not in (None, "ANY"):
                base = base[base["Club"] == club]

            if base.empty:
                base = df
            result_df = self._apply_filters_and_sort(base, spec, quantile_base=base)

        results = [self._format_row(row, spec) for _, row in result_df.iterrows()]
        candidate_names = result_df["Player Name"].tolist()
        return results, candidate_names
