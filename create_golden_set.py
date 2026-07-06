"""
골든셋 생성 스크립트
단발형 20개 / 복합형 15개 / followup 15개 / refusal 10개 = 총 60개
"""

import pandas as pd
import json
import os

PROCESSED_PATH = "data/processed/players_clean.csv"
OUTPUT_PATH = "data/golden_set.json"

os.makedirs("data", exist_ok=True)


def load(path: str = PROCESSED_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# ── 단발형 20개 ───────────────────────────────────────────
SINGLE_TEMPLATES = [
    # MID
    {"id": "S001", "question": "이번 시즌 인터셉트가 가장 많은 미드필더는?",
     "position": "MID", "metric": "Interceptions", "top_n": 3},
    {"id": "S002", "question": "태클 횟수가 가장 많은 미드필더는?",
     "position": "MID", "metric": "Tackles", "top_n": 3},
    {"id": "S003", "question": "프로그레시브 캐리가 가장 많은 미드필더는?",
     "position": "MID", "metric": "Progressive Carries", "top_n": 3},
    {"id": "S004", "question": "어시스트가 가장 많은 미드필더는?",
     "position": "MID", "metric": "Assists", "top_n": 3},
    {"id": "S005", "question": "패스 성공률이 가장 높은 미드필더는?",
     "position": "MID", "metric": "Passes%", "top_n": 3},
    {"id": "S006", "question": "이번 시즌 가장 많이 뛴 미드필더는?",
     "position": "MID", "metric": "Minutes", "top_n": 3},
    {"id": "S007", "question": "볼 점유 획득 횟수가 가장 많은 미드필더는?",
     "position": "MID", "metric": "Possession Won", "top_n": 3},
    {"id": "S008", "question": "골을 가장 많이 넣은 미드필더는?",
     "position": "MID", "metric": "Goals", "top_n": 3},
    # DEF
    {"id": "S009", "question": "태클 횟수가 가장 많은 수비수는?",
     "position": "DEF", "metric": "Tackles", "top_n": 3},
    {"id": "S010", "question": "인터셉트가 가장 많은 수비수는?",
     "position": "DEF", "metric": "Interceptions", "top_n": 3},
    {"id": "S011", "question": "클리어런스가 가장 많은 수비수는?",
     "position": "DEF", "metric": "Clearances", "top_n": 3},
    {"id": "S012", "question": "공중 듀얼 승률이 가장 높은 수비수는?",
     "position": "DEF", "metric": "aDuels %", "top_n": 3},
    {"id": "S013", "question": "패스 성공률이 가장 높은 수비수는?",
     "position": "DEF", "metric": "Passes%", "top_n": 3},
    # FWD
    {"id": "S014", "question": "골을 가장 많이 넣은 공격수는?",
     "position": "FWD", "metric": "Goals", "top_n": 3},
    {"id": "S015", "question": "슈팅 성공률이 가장 높은 공격수는?",
     "position": "FWD", "metric": "Conversion %", "top_n": 3},
    {"id": "S016", "question": "어시스트가 가장 많은 공격수는?",
     "position": "FWD", "metric": "Assists", "top_n": 3},
    {"id": "S017", "question": "프로그레시브 캐리가 가장 많은 공격수는?",
     "position": "FWD", "metric": "Progressive Carries", "top_n": 3},
    # GKP
    {"id": "S018", "question": "이번 시즌 가장 많이 뛴 골키퍼는?",
     "position": "GKP", "metric": "Minutes", "top_n": 3},
    {"id": "S019", "question": "세이브 횟수가 가장 많은 골키퍼는?",
     "position": "GKP", "metric": "Saves", "top_n": 3},
    {"id": "S020", "question": "클린시트가 가장 많은 골키퍼는?",
     "position": "GKP", "metric": "Clean Sheets", "top_n": 3},
]


def build_single(df: pd.DataFrame) -> list[dict]:
    results = []
    for t in SINGLE_TEMPLATES:
        filtered = df[df["Position"] == t["position"]]
        top = filtered.nlargest(t["top_n"], t["metric"])["Player Name"].tolist()
        results.append({
            "id": t["id"],
            "type": "single",
            "question": t["question"],
            "expected_players": top,
            "position_filter": t["position"],
            "metric": t["metric"],
        })
    return results


# ── 복합형 15개 ───────────────────────────────────────────
def build_complex(df: pd.DataFrame) -> list[dict]:
    results = []

    # C001
    sub = df[(df["Position"] == "MID") & (df["Passes%"] >= 80)]
    top = sub.nlargest(3, "Tackles")["Player Name"].tolist()
    results.append({
        "id": "C001", "type": "complex",
        "question": "패스 성공률 80% 이상이면서 태클도 많은 미드필더는?",
        "expected_players": top,
        "conditions": {"position": "MID", "Passes%_min": 80, "sort_by": "Tackles"},
    })

    # C002
    sub = df[(df["Position"] == "MID") & (df["Interceptions"] >= df[df["Position"] == "MID"]["Interceptions"].quantile(0.5))]
    top = sub.nlargest(3, "Progressive Carries")["Player Name"].tolist()
    results.append({
        "id": "C002", "type": "complex",
        "question": "수비 기여도(인터셉트)도 높으면서 전진 능력(프로그레시브 캐리)도 좋은 미드필더는?",
        "expected_players": top,
        "conditions": {"position": "MID", "Interceptions_min_percentile": 50, "sort_by": "Progressive Carries"},
    })

    # C003
    sub = df[(df["Position"] == "DEF") & (df["Tackles"] >= df[df["Position"] == "DEF"]["Tackles"].quantile(0.5))]
    top = sub.nlargest(3, "Interceptions")["Player Name"].tolist()
    results.append({
        "id": "C003", "type": "complex",
        "question": "태클과 인터셉트가 모두 높은 수비수는?",
        "expected_players": top,
        "conditions": {"position": "DEF", "Tackles_min_percentile": 50, "sort_by": "Interceptions"},
    })

    # C004
    sub = df[(df["Position"] == "FWD") & (df["Goals"] >= 5)]
    top = sub.nlargest(3, "Assists")["Player Name"].tolist()
    results.append({
        "id": "C004", "type": "complex",
        "question": "골도 5개 이상이면서 어시스트도 많은 공격수는?",
        "expected_players": top,
        "conditions": {"position": "FWD", "Goals_min": 5, "sort_by": "Assists"},
    })

    # C005
    sub = df[(df["Position"] == "MID") & (df["Minutes"] >= 1800)]
    top = sub.nlargest(3, "Goals")["Player Name"].tolist()
    results.append({
        "id": "C005", "type": "complex",
        "question": "주전(1800분 이상)으로 뛰면서 골도 많이 넣은 미드필더는?",
        "expected_players": top,
        "conditions": {"position": "MID", "Minutes_min": 1800, "sort_by": "Goals"},
    })

    # C006
    sub = df[(df["Position"] == "DEF") & (df["Passes%"] >= 85)]
    top = sub.nlargest(3, "Progressive Carries")["Player Name"].tolist()
    results.append({
        "id": "C006", "type": "complex",
        "question": "패스 성공률 85% 이상의 빌드업 능력 있는 수비수는?",
        "expected_players": top,
        "conditions": {"position": "DEF", "Passes%_min": 85, "sort_by": "Progressive Carries"},
    })

    # C007
    # 원래 quantile(0.6)이었는데, 파서 규칙("명시적 숫자 없으면 항상 50% 기본값")과 어긋나서
    # 다른 percentile 항목들처럼 50%로 통일.
    sub = df[(df["Position"] == "MID") & (df["Tackles"] >= df[df["Position"] == "MID"]["Tackles"].quantile(0.5))]
    top = sub.nlargest(3, "Passes%")["Player Name"].tolist()
    results.append({
        "id": "C007", "type": "complex",
        "question": "압박과 태클이 강하면서 패스 정확도도 높은 미드필더는?",
        "expected_players": top,
        "conditions": {"position": "MID", "Tackles_min_percentile": 50, "sort_by": "Passes%"},
    })

    # C008
    sub = df[(df["Position"] == "FWD") & (df["Shots"] >= df[df["Position"] == "FWD"]["Shots"].quantile(0.5))]
    top = sub.nlargest(3, "Conversion %")["Player Name"].tolist()
    results.append({
        "id": "C008", "type": "complex",
        "question": "슈팅 시도도 많으면서 전환율도 높은 공격수는?",
        "expected_players": top,
        "conditions": {"position": "FWD", "Shots_min_percentile": 50, "sort_by": "Conversion %"},
    })

    # C009
    # 원래 quantile(0.4)였는데, 파서 규칙("명시적 숫자 없으면 항상 50% 기본값")과 어긋나서
    # 다른 percentile 항목들처럼 50%로 통일.
    sub = df[(df["Position"] == "MID") & (df["Fouls"] <= df[df["Position"] == "MID"]["Fouls"].quantile(0.5))]
    top = sub.nlargest(3, "Tackles")["Player Name"].tolist()
    results.append({
        "id": "C009", "type": "complex",
        "question": "파울 없이 클린하게 수비하면서 태클 횟수도 많은 미드필더는?",
        "expected_players": top,
        "conditions": {"position": "MID", "Fouls_max_percentile": 50, "sort_by": "Tackles"},
    })

    # C010
    sub = df[(df["Position"] == "GKP") & (df["Minutes"] >= 1800)]
    top = sub.nlargest(3, "Saves %")["Player Name"].tolist()
    results.append({
        "id": "C010", "type": "complex",
        "question": "주전 골키퍼 중 세이브 성공률이 높은 선수는?",
        "expected_players": top,
        "conditions": {"position": "GKP", "Minutes_min": 1800, "sort_by": "Saves %"},
    })

    # C011
    sub = df[(df["Position"] == "DEF") & (df["Aerial Duels"] >= df[df["Position"] == "DEF"]["Aerial Duels"].quantile(0.5))]
    top = sub.nlargest(3, "aDuels %")["Player Name"].tolist()
    results.append({
        "id": "C011", "type": "complex",
        "question": "공중 듀얼 참여도 많고 승률도 높은 수비수는?",
        "expected_players": top,
        "conditions": {"position": "DEF", "Aerial Duels_min_percentile": 50, "sort_by": "aDuels %"},
    })

    # C012
    sub = df[(df["Position"] == "MID") & (df["Assists"] >= 3)]
    top = sub.nlargest(3, "Passes%")["Player Name"].tolist()
    results.append({
        "id": "C012", "type": "complex",
        "question": "어시스트 3개 이상이면서 패스 성공률도 높은 창의적인 미드필더는?",
        "expected_players": top,
        "conditions": {"position": "MID", "Assists_min": 3, "sort_by": "Passes%"},
    })

    # C013
    sub = df[(df["Position"] == "FWD") & (df["Progressive Carries"] >= df[df["Position"] == "FWD"]["Progressive Carries"].quantile(0.5))]
    top = sub.nlargest(3, "Assists")["Player Name"].tolist()
    results.append({
        "id": "C013", "type": "complex",
        "question": "직접 전진하면서 어시스트도 많이 만들어내는 윙어는?",
        "expected_players": top,
        "conditions": {"position": "FWD", "Progressive Carries_min_percentile": 50, "sort_by": "Assists"},
    })

    # C014
    # 원래 Yellow Cards<=3 절대값으로 하드코딩돼 있었는데, "카드 없이/파울 없이" 패턴은
    # 다른 항목들(C009 등)처럼 percentile 필터로 통일해야 파서 규칙과 golden 정답이 일치한다.
    mid_pool = df[df["Position"] == "MID"]
    sub = df[
        (df["Position"] == "MID")
        & (df["Yellow Cards"] <= mid_pool["Yellow Cards"].quantile(0.5))
        & (df["Minutes"] >= 1800)
    ]
    top = sub.nlargest(3, "Tackles")["Player Name"].tolist()
    results.append({
        "id": "C014", "type": "complex",
        "question": "주전으로 뛰면서 경고 카드 없이 강한 압박을 하는 미드필더는?",
        "expected_players": top,
        "conditions": {"position": "MID", "Yellow Cards_max_percentile": 50, "Minutes_min": 1800, "sort_by": "Tackles"},
    })

    # C015
    sub = df[(df["Position"] == "DEF") & (df["Blocks"] >= df[df["Position"] == "DEF"]["Blocks"].quantile(0.5))]
    top = sub.nlargest(3, "Clearances")["Player Name"].tolist()
    results.append({
        "id": "C015", "type": "complex",
        "question": "블록과 클리어런스 모두 많은 헌신적인 수비수는?",
        "expected_players": top,
        "conditions": {"position": "DEF", "Blocks_min_percentile": 50, "sort_by": "Clearances"},
    })

    return results


# ── followup 15개 ─────────────────────────────────────────
def build_followup(df: pd.DataFrame) -> list[dict]:

    def top(pos, metric, n=3):
        return df[df["Position"] == pos].nlargest(n, metric)["Player Name"].tolist()

    def safe_top(players, metric, n=1, ascending=False):
        sub = df[df["Player Name"].isin(players)]
        if sub.empty:
            return players[:n]
        result = (sub.nsmallest(n, metric) if ascending else sub.nlargest(n, metric))["Player Name"].tolist()
        return result if result else players[:n]

    return [
        {
            "id": "F001", "type": "followup",
            "conversation": [
                {"turn": 1, "question": "태클 많은 미드필더 3명 추천해줘",
                 "expected_players": top("MID", "Tackles")},
                {"turn": 2, "question": "그 중에서 패스 성공률이 제일 높은 선수는?",
                 "expected_players": safe_top(top("MID", "Tackles"), "Passes%")},
            ]
        },
        {
            "id": "F002", "type": "followup",
            "conversation": [
                {"turn": 1, "question": "골 많이 넣은 공격수 알려줘",
                 "expected_players": top("FWD", "Goals")},
                {"turn": 2, "question": "그 선수들 소속팀이 어디야?",
                 "expected_players": top("FWD", "Goals")},
                {"turn": 3, "question": "그 중 슈팅 성공률이 가장 높은 선수는?",
                 "expected_players": safe_top(top("FWD", "Goals"), "Conversion %")},
            ]
        },
        {
            "id": "F003", "type": "followup",
            "conversation": [
                {"turn": 1, "question": "인터셉트 많은 수비수 추천해줘",
                 "expected_players": top("DEF", "Interceptions")},
                {"turn": 2, "question": "그 중 공중 듀얼 승률이 높은 선수는?",
                 "expected_players": safe_top(top("DEF", "Interceptions"), "aDuels %")},
            ]
        },
        {
            "id": "F004", "type": "followup",
            "conversation": [
                {"turn": 1, "question": "주전 골키퍼 중 세이브 많이 한 선수는?",
                 "expected_players": df[(df["Position"] == "GKP") & (df["Minutes"] >= 1800)].nlargest(3, "Saves")["Player Name"].tolist()},
                {"turn": 2, "question": "그 선수 국적이 어떻게 돼?",
                 "expected_players": df[(df["Position"] == "GKP") & (df["Minutes"] >= 1800)].nlargest(3, "Saves")["Player Name"].tolist()},
            ]
        },
        {
            "id": "F005", "type": "followup",
            "conversation": [
                {"turn": 1, "question": "어시스트 많은 미드필더 알려줘",
                 "expected_players": top("MID", "Assists")},
                {"turn": 2, "question": "그 중 프로그레시브 캐리도 많은 선수는?",
                 "expected_players": safe_top(top("MID", "Assists"), "Progressive Carries")},
                {"turn": 3, "question": "그 선수 이번 시즌 몇 분 뛰었어?",
                 "expected_players": safe_top(top("MID", "Assists"), "Progressive Carries")},
            ]
        },
        {
            "id": "F006", "type": "followup",
            "conversation": [
                {"turn": 1, "question": "패스 성공률 높은 수비수 추천해줘",
                 "expected_players": top("DEF", "Passes%")},
                {"turn": 2, "question": "그 중 가장 많이 뛴 선수는?",
                 "expected_players": safe_top(top("DEF", "Passes%"), "Minutes")},
            ]
        },
        {
            "id": "F007", "type": "followup",
            "conversation": [
                {"turn": 1, "question": "프로그레시브 캐리 많은 미드필더 알려줘",
                 "expected_players": top("MID", "Progressive Carries")},
                {"turn": 2, "question": "그 중 수비 기여도(인터셉트)도 높은 선수가 있어?",
                 "expected_players": safe_top(top("MID", "Progressive Carries"), "Interceptions")},
            ]
        },
        {
            "id": "F008", "type": "followup",
            "conversation": [
                {"turn": 1, "question": "클리어런스 많은 수비수 3명 알려줘",
                 "expected_players": top("DEF", "Clearances")},
                {"turn": 2, "question": "그 중 태클도 많이 하는 선수는?",
                 "expected_players": safe_top(top("DEF", "Clearances"), "Tackles")},
                {"turn": 3, "question": "그 선수 소속팀은 어디야?",
                 "expected_players": safe_top(top("DEF", "Clearances"), "Tackles")},
            ]
        },
        {
            "id": "F009", "type": "followup",
            "conversation": [
                {"turn": 1, "question": "슈팅 많이 하는 공격수 추천해줘",
                 "expected_players": top("FWD", "Shots")},
                {"turn": 2, "question": "그 중 유효슈팅 비율도 높은 선수는?",
                 "expected_players": safe_top(top("FWD", "Shots"), "Shots On Target")},
            ]
        },
        {
            "id": "F010", "type": "followup",
            "conversation": [
                {"turn": 1, "question": "볼 점유 획득이 많은 미드필더는?",
                 "expected_players": top("MID", "Possession Won")},
                {"turn": 2, "question": "그 선수들 평균 출전 시간은 어떻게 돼?",
                 "expected_players": top("MID", "Possession Won")},
            ]
        },
        {
            "id": "F011", "type": "followup",
            "conversation": [
                {"turn": 1, "question": "태클 강한 수비수 알려줘",
                 "expected_players": top("DEF", "Tackles")},
                {"turn": 2, "question": "그 중 경고 카드 적은 선수는?",
                 "expected_players": safe_top(top("DEF", "Tackles"), "Yellow Cards", ascending=True)},
            ]
        },
        {
            "id": "F012", "type": "followup",
            "conversation": [
                {"turn": 1, "question": "이번 시즌 가장 많이 뛴 미드필더는?",
                 "expected_players": top("MID", "Minutes")},
                {"turn": 2, "question": "그 선수 골+어시스트 합산은?",
                 "expected_players": top("MID", "Minutes")[:1]},
            ]
        },
        {
            "id": "F013", "type": "followup",
            "conversation": [
                {"turn": 1, "question": "세이브율 높은 골키퍼 추천해줘",
                 "expected_players": top("GKP", "Saves %")},
                {"turn": 2, "question": "그 중 클린시트도 많은 선수는?",
                 "expected_players": safe_top(top("GKP", "Saves %"), "Clean Sheets")},
            ]
        },
        {
            "id": "F014", "type": "followup",
            "conversation": [
                {"turn": 1, "question": "크로스 성공률 높은 공격수는?",
                 "expected_players": df[df["Position"] == "FWD"].nlargest(3, "Crosses %")["Player Name"].tolist()},
                {"turn": 2, "question": "그 선수들 어시스트는 몇 개야?",
                 "expected_players": df[df["Position"] == "FWD"].nlargest(3, "Crosses %")["Player Name"].tolist()},
            ]
        },
        {
            "id": "F015", "type": "followup",
            "conversation": [
                {"turn": 1, "question": "블록 많이 하는 수비수 알려줘",
                 "expected_players": top("DEF", "Blocks")},
                {"turn": 2, "question": "그 중 패스 성공률도 좋은 선수는?",
                 "expected_players": safe_top(top("DEF", "Blocks"), "Passes%")},
                {"turn": 3, "question": "그 선수 국적이 어디야?",
                 "expected_players": safe_top(top("DEF", "Blocks"), "Passes%")},
            ]
        },
    ]


# ── refusal 10개 ──────────────────────────────────────────
REFUSAL_SET = [
    {"id": "R001", "type": "refusal",
     "question": "음바페 이번 시즌 PL 성적 알려줘",
     "expected_response": "refusal",
     "reason": "PL 소속 선수 아님"},
    {"id": "R002", "type": "refusal",
     "question": "손흥민 챔피언스리그 스탯은?",
     "expected_response": "refusal",
     "reason": "데이터에 챔피언스리그 스탯 없음 (PL 스탯만 존재)"},
    {"id": "R003", "type": "refusal",
     "question": "2023-24 시즌 최고의 미드필더는?",
     "expected_response": "refusal",
     "reason": "보유 데이터는 2024-25 시즌만 포함"},
    {"id": "R004", "type": "refusal",
     "question": "메시 PL 통산 기록 알려줘",
     "expected_response": "refusal",
     "reason": "메시는 PL 소속 이력 없음"},
    {"id": "R005", "type": "refusal",
     "question": "이번 시즌 PL 우승팀은 어디야?",
     "expected_response": "refusal",
     "reason": "팀 성적/순위 데이터 없음 (선수 스탯만 존재)"},
    {"id": "R006", "type": "refusal",
     "question": "음바페 주급이 얼마야?",
     "expected_response": "refusal",
     "reason": "연봉/계약 데이터 없음"},
    {"id": "R007", "type": "refusal",
     "question": "라리가 최고의 미드필더 추천해줘",
     "expected_response": "refusal",
     "reason": "보유 데이터는 PL만 포함"},
    {"id": "R008", "type": "refusal",
     "question": "홀란드 부상 이력 알려줘",
     "expected_response": "refusal",
     "reason": "부상 관련 데이터 없음"},
    {"id": "R009", "type": "refusal",
     "question": "네이마르 이번 시즌 스탯은?",
     "expected_response": "refusal",
     "reason": "PL 소속 선수 아님"},
    {"id": "R010", "type": "refusal",
     "question": "PL 역대 최다 득점자는 누구야?",
     "expected_response": "refusal",
     "reason": "역대 통산 기록 데이터 없음 (24-25 시즌만 존재)"},
]


# ── 메인 실행 ─────────────────────────────────────────────
def run():
    df = load()

    single = build_single(df)
    complex_ = build_complex(df)
    followup = build_followup(df)
    refusal = REFUSAL_SET

    golden_set = {
        "meta": {
            "total": len(single) + len(complex_) + len(followup) + len(refusal),
            "single": len(single),
            "complex": len(complex_),
            "followup": len(followup),
            "refusal": len(refusal),
            "data_source": "epl_player_stats_24_25.csv",
            "season": "2024-25",
        },
        "data": single + complex_ + followup + refusal,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(golden_set, f, ensure_ascii=False, indent=2)

    print(f"✅ 골든셋 생성 완료: 총 {golden_set['meta']['total']}개")
    print(f"   단발형 {len(single)}개 / 복합형 {len(complex_)}개 / followup {len(followup)}개 / refusal {len(refusal)}개")
    print(f"   저장 위치: {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
