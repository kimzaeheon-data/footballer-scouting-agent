import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import MinMaxScaler

# ── 경로 설정 ──────────────────────────────────────────────
RAW_PATH = "data/epl_player_stats_24_25.csv"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── 포지션 그룹 매핑 ───────────────────────────────────────
POSITION_MAP = {
    "GKP": "GKP",
    "DEF": "DEF",
    "MID": "MID",
    "FWD": "FWD",
}

# 포지션 정규화 함수 (첫 번째 포지션만 사용)
def normalize_position(pos: str) -> str:
    if pd.isna(pos):
        return "Unknown"
    primary = str(pos).split(",")[0].split("/")[0].strip().upper()
    return POSITION_MAP.get(primary, primary)


# ── 1. 데이터 로딩 ─────────────────────────────────────────
def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load] {df.shape[0]}명 로드 완료")
    return df


# 최소 출전 시간 기준: "최소 20경기(1800분) 이상 출전"을 기본으로 하되, 특정 포지션 표본이
# 너무 작아지면(스카우팅에 쓸 만한 최소 인원 미만) 15경기(1350분)로 완화한다.
MIN_MINUTES_PRIMARY = 1800
MIN_MINUTES_FALLBACK = 1350
MIN_GROUP_SIZE = 15  # 포지션별 최소 표본 수


# ── 2. 결측치 처리 ─────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 포지션 정규화
    df["Position"] = df["Position"].apply(normalize_position)

    # % 컬럼 → float 변환
    pct_cols = [c for c in df.columns if "%" in c]
    for col in pct_cols:
        df[col] = df[col].astype(str).str.replace("%", "").str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 수치형 결측치 → 0 (출전 없는 선수는 0이 맞음)
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(0)

    # 문자형 결측치 → "Unknown"
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].fillna("Unknown")

    # 최소 출전 시간 필터: 1800분(20경기) 기준으로 걸러보고, 어느 포지션이라도 표본이
    # MIN_GROUP_SIZE 밑으로 떨어지면 전체를 1350분(15경기) 기준으로 완화한다.
    primary = df[df["Minutes"] >= MIN_MINUTES_PRIMARY]
    group_sizes = primary["Position"].value_counts()
    if not group_sizes.empty and group_sizes.min() < MIN_GROUP_SIZE:
        threshold = MIN_MINUTES_FALLBACK
        print(
            f"[clean] {MIN_MINUTES_PRIMARY}분 기준 시 최소 포지션 표본이 {group_sizes.min()}명"
            f"(< {MIN_GROUP_SIZE})이라 {MIN_MINUTES_FALLBACK}분으로 완화"
        )
    else:
        threshold = MIN_MINUTES_PRIMARY

    df = df[df["Minutes"] >= threshold].reset_index(drop=True)

    print(f"[clean] 출전시간 {threshold}분 이상 필터링 후 {df.shape[0]}명")
    return df


# ── 3. 정규화 (90분당 스탯 + MinMax) ─────────────────────
PER90_COLS = [
    "Goals", "Assists", "Shots", "Shots On Target",
    "Passes", "Successful Passes", "Through Balls",
    "Progressive Carries", "Possession Won", "Interceptions",
    "Blocks", "Tackles", "Clearances", "Fouls",
    "Yellow Cards", "Red Cards",
]

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    minutes = df["Minutes"].replace(0, np.nan)

    # 90분당 스탯 생성
    for col in PER90_COLS:
        if col in df.columns:
            df[f"{col}_p90"] = (df[col] / minutes * 90).round(3)

    # MinMax 정규화 (p90 컬럼만)
    p90_cols = [c for c in df.columns if c.endswith("_p90")]
    scaler = MinMaxScaler()
    df[p90_cols] = scaler.fit_transform(df[p90_cols].fillna(0)).round(4)

    print(f"[normalize] p90 컬럼 {len(p90_cols)}개 정규화 완료")
    return df


# ── 4. RAG 청킹 ───────────────────────────────────────────
def chunk_for_rag(df: pd.DataFrame) -> list[dict]:
    """선수 1명 = 청크 1개 (자연어 + 메타데이터)"""
    chunks = []
    for _, row in df.iterrows():
        text = (
            f"{row['Player Name']}은(는) {row['Club']} 소속 {row['Position']} 포지션 선수다. "
            f"국적은 {row['Nationality']}이며 이번 시즌 {int(row['Appearances'])}경기에 출전해 "
            f"{int(row['Minutes'])}분을 소화했다. "
            f"골 {int(row['Goals'])}개, 어시스트 {int(row['Assists'])}개를 기록했으며 "
            f"패스 성공률은 {row['Passes%']}%다. "
            f"태클 {int(row['Tackles'])}회, 인터셉트 {int(row['Interceptions'])}회, "
            f"프로그레시브 캐리 {int(row['Progressive Carries'])}회를 기록했다."
        )

        meta = {
            "player": row["Player Name"],
            "club": row["Club"],
            "position": row["Position"],
            "nationality": row["Nationality"],
            "minutes": int(row["Minutes"]),
            "goals": int(row["Goals"]),
            "assists": int(row["Assists"]),
            "passes_pct": float(row["Passes%"]),
            "tackles": int(row["Tackles"]),
            "interceptions": int(row["Interceptions"]),
            "progressive_carries": int(row["Progressive Carries"]),
        }

        chunks.append({"text": text, "metadata": meta})

    print(f"[chunk] 청크 {len(chunks)}개 생성 완료")
    return chunks



# ── 5. 전체 파이프라인 실행 ───────────────────────────────
def run():
    df_raw = load_data()
    df_clean = clean_data(df_raw)
    df_norm = normalize_data(df_clean)

    # 저장
    df_norm.to_csv(f"{PROCESSED_DIR}/players_clean.csv", index=False)
    print(f"[save] players_clean.csv 저장 완료")

    chunks = chunk_for_rag(df_clean)
    with open(f"{PROCESSED_DIR}/players_chunked.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[save] players_chunked.json 저장 완료")
    print("\n✅ 전처리 파이프라인 완료")
    return df_norm, chunks


if __name__ == "__main__":
    run()
