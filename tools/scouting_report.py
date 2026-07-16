"""
선수 스카우팅 리포트 이미지 카드 생성기.

router_rag가 답변에서 선수를 추천한 뒤 "스카우팅 리포트 작성해드릴까요?"라고 제안하고,
사용자가 승낙하면 이 모듈로 이미지 카드를 만든다.

주의: 이 데이터셋(players_clean.csv)엔 슈팅/터치 좌표 같은 위치 데이터가 없어서 실제 "히트맵"
(경기장 위 움직임 분포)은 만들 수 없다. 대신 포지션 동료들 사이에서의 퍼센타일 랭킹을 시각화한다.
서술형 코멘트도 LLM이 자유 생성하는 대신, 실제 랭킹 숫자를 그대로 문장에 꽂아 넣는 템플릿 방식을
쓴다 — 근거 없는 문장(hallucination)을 원천적으로 막기 위함.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as patches

PROCESSED_PATH = "data/processed/players_clean.csv"

# 나눔/Noto CJK 폰트 등록 (한글 깨짐 방지)
# 파일 경로로 등록 시도 (플랫폼별 실제 폰트 파일 위치) → 실패하면 시스템에 이미 등록된
# 폰트 패밀리명으로 폴백. 이전엔 Linux 경로(Cowork 샌드박스)만 있어서 로컬 macOS에서
# 아무 것도 안 걸리고 조용히 기본 폰트(한글 미지원)로 떨어져 글자가 깨졌었음.
_CJK_FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux (Cowork 샌드박스 등)
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # Linux (배포판에 따라 경로 다름)
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",               # macOS
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",       # macOS (구버전 대비)
    "C:/Windows/Fonts/malgun.ttf",                                # Windows
]
_CJK_FAMILY_FALLBACKS = [
    "Apple SD Gothic Neo", "AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR",
]
_FONT_PROP = None
for _p in _CJK_FONT_PATHS:
    if os.path.exists(_p):
        fm.fontManager.addfont(_p)
        _FONT_PROP = fm.FontProperties(fname=_p)
        plt.rcParams["font.family"] = _FONT_PROP.get_name()
        break

if _FONT_PROP is None:
    _available = {f.name for f in fm.fontManager.ttflist}
    for _fam in _CJK_FAMILY_FALLBACKS:
        if _fam in _available:
            plt.rcParams["font.family"] = _fam
            break

POSITION_LABEL = {"FWD": "공격수", "MID": "미드필더", "DEF": "수비수", "GKP": "골키퍼"}
POSITION_COLOR = {"FWD": "#c0392b", "MID": "#1a6b3c", "DEF": "#1f4e8c", "GKP": "#8c6d1f"}

# 포지션별로 보여줄 핵심 지표 10개(2행 x 5열) — 컬럼명, 표시 약어, 정렬 방향(True=높을수록 좋음)
POSITION_STAT_SETS = {
    "FWD": [
        ("Goals", "GO", True), ("Shots On Target", "ST", True), ("Conversion %", "CV", True),
        ("Big Chances Missed", "BC", False), ("Offsides", "OF", False),
        ("Assists", "AS", True), ("Progressive Carries", "PC", True), ("Touches", "TC", True),
        ("Dispossessed", "DP", False), ("Carries Ended with Chance", "CC", True),
    ],
    "MID": [
        ("Tackles", "TK", True), ("Interceptions", "IC", True), ("Possession Won", "PW", True),
        ("Dispossessed", "DP", False), ("Fouls", "FS", False),
        ("Assists", "AS", True), ("Progressive Carries", "PC", True), ("Passes%", "P%", True),
        ("Crosses %", "CR", True), ("Through Balls", "TB", True),
    ],
    "DEF": [
        ("Tackles", "TK", True), ("Interceptions", "IC", True), ("Clearances", "CL", True),
        ("Blocks", "BK", True), ("Fouls", "FS", False),
        ("aDuels %", "AD", True), ("Passes%", "P%", True), ("Progressive Carries", "PC", True),
        ("Crosses %", "CR", True), ("Own Goals", "OG", False),
    ],
    "GKP": [
        ("Saves", "SV", True), ("Saves %", "S%", True), ("Clean Sheets", "CS", True),
        ("Goals Prevented", "GP", True), ("High Claims", "HC", True),
        ("Punches", "PU", True), ("Goals Conceded", "GC", False), ("xGoT Conceded", "XG", False),
        ("Passes%", "P%", True), ("Own Goals", "OG", False),
    ],
}

KOREAN_STAT_NAME = {
    "Goals": "골", "Shots On Target": "유효슈팅", "Conversion %": "결정력",
    "Big Chances Missed": "빅찬스실패", "Offsides": "오프사이드", "Assists": "도움",
    "Progressive Carries": "전진드리블", "Touches": "볼터치", "Dispossessed": "볼뺏김",
    "Carries Ended with Chance": "찬스연결",
    "Tackles": "태클", "Interceptions": "인터셉트", "Possession Won": "볼경합승리",
    "Fouls": "파울", "Passes%": "패스성공률", "Crosses %": "크로스성공률",
    "Through Balls": "스루패스",
    "Clearances": "클리어런스", "Blocks": "슈팅차단", "aDuels %": "공중볼경합률",
    "Own Goals": "자책골",
    "Saves": "선방", "Saves %": "선방률", "Clean Sheets": "무실점",
    "Goals Prevented": "실점방지", "High Claims": "하이볼처리", "Punches": "펀칭",
    "Goals Conceded": "실점", "xGoT Conceded": "유효슈팅실점기대값",
}

PITCH_POSITION_XY = {
    "GKP": (0.5, 0.08),
    "DEF": (0.5, 0.3),
    "MID": (0.5, 0.58),
    "FWD": (0.5, 0.85),
}


def _load_df():
    return pd.read_csv(PROCESSED_PATH)


def find_player(player_name: str, df: pd.DataFrame = None):
    """이름 부분일치로 선수 row 찾기 (없으면 None)."""
    df = df if df is not None else _load_df()
    mask = df["Player Name"].str.contains(player_name, case=False, na=False, regex=False)
    matches = df[mask]
    if matches.empty:
        return None
    return matches.iloc[0]


def compute_rankings(player_row: pd.Series, df: pd.DataFrame):
    """포지션 동료들 사이에서 각 핵심 지표의 순위/백분위를 계산."""
    position = player_row["Position"]
    peers = df[df["Position"] == position]
    n_peers = len(peers)
    stat_set = POSITION_STAT_SETS.get(position, POSITION_STAT_SETS["MID"])

    rankings = []
    for col, abbr, higher_is_better in stat_set:
        if col not in df.columns:
            continue
        ranked = peers[col].rank(ascending=not higher_is_better, method="min")
        player_idx = peers.index[peers["Player Name"] == player_row["Player Name"]]
        if len(player_idx) == 0:
            continue
        rank = int(ranked.loc[player_idx[0]])
        percentile = 100 * (1 - (rank - 1) / max(n_peers - 1, 1))
        rankings.append({
            "column": col, "abbr": abbr, "value": player_row[col],
            "rank": rank, "n_peers": n_peers, "percentile": round(percentile, 1),
            "higher_is_better": higher_is_better,
        })
    return rankings, n_peers


def _top_strengths_text(player_row: pd.Series, rankings: list, top_k: int = 3) -> str:
    """랭킹 상위 지표를 골라 근거 기반 문장 템플릿으로 조립 (LLM 프리텍스트 없음, 숫자만 그대로 인용).

    "적을수록 좋은" 지표(자책골/경고 등)는 0에서 동률이 흔해 "1위"가 남발되고 실제로는
    자랑거리가 아니므로 강점 후보에서 제외 — 실제로 Own Goals=0 동률 다수가 rank=1로 잡혀
    무의미한 "1위" 문구가 나오는 걸 확인하고 반영함.
    """
    name = player_row["Player Name"]
    position_kr = POSITION_LABEL.get(player_row["Position"], player_row["Position"])
    positive_rankings = [r for r in rankings if r["higher_is_better"]]
    sorted_by_rank = sorted(positive_rankings, key=lambda r: r["rank"])
    strengths = sorted_by_rank[:top_k]

    lines = [f"{name} ({position_kr}, {player_row['Club']})"]
    parts = []
    for s in strengths:
        top_percent = max(1, round(100 * s["rank"] / s["n_peers"]))
        parts.append(f"{s['abbr']}({s['column']}) 부문 포지션 내 {s['rank']}위/{s['n_peers']}명 "
                      f"(상위 {top_percent}%)")
    lines.append("이번 시즌 " + ", ".join(parts) + "을 기록했다.")
    lines.append(f"출전 {int(player_row['Appearances'])}경기, {int(player_row['Minutes'])}분 소화, "
                 f"골 {int(player_row['Goals'])} / 도움 {int(player_row['Assists'])} "
                 f"/ 경고 {int(player_row['Yellow Cards'])} / 퇴장 {int(player_row['Red Cards'])}.")
    return "\n".join(lines)


def render_scouting_card(player_name: str, output_path: str = None, df: pd.DataFrame = None) -> str:
    """선수 이름으로 스카우팅 리포트 이미지 카드를 만들어 파일로 저장하고 경로를 반환."""
    df = df if df is not None else _load_df()
    row = find_player(player_name, df)
    if row is None:
        raise ValueError(f"'{player_name}' 선수를 데이터에서 찾을 수 없습니다.")

    position = row["Position"]
    color = POSITION_COLOR.get(position, "#333333")
    rankings, n_peers = compute_rankings(row, df)

    fig = plt.figure(figsize=(6.4, 9.0), dpi=150)
    fig.patch.set_facecolor("white")

    # ── 헤더 ──
    ax_header = fig.add_axes([0.04, 0.90, 0.92, 0.08])
    ax_header.set_xlim(0, 1); ax_header.set_ylim(0, 1); ax_header.axis("off")
    ax_header.add_patch(patches.FancyBboxPatch((0, 0), 0.16, 1, boxstyle="round,pad=0.01",
                                                linewidth=1.5, edgecolor=color, facecolor="white"))
    ax_header.text(0.08, 0.62, position, ha="center", va="center", fontsize=13, fontweight="bold", color=color)
    ax_header.add_patch(patches.Rectangle((0.19, 0), 0.77, 1, linewidth=1.5, edgecolor=color, facecolor="white"))
    ax_header.text(0.22, 0.6, row["Player Name"], fontsize=17, fontweight="bold", va="center")
    ax_header.text(0.22, 0.18, f"{row['Club']} · {row.get('Nationality', '')}", fontsize=10, va="center", color="#555")

    # ── 아이콘 행 (출전경기/시간/골/도움/경고/퇴장) ──
    ax_icons = fig.add_axes([0.04, 0.83, 0.92, 0.055])
    ax_icons.set_xlim(0, 1); ax_icons.set_ylim(0, 1); ax_icons.axis("off")
    icon_stats = [
        ("경기", int(row["Appearances"])), ("분", int(row["Minutes"])), ("골", int(row["Goals"])),
        ("도움", int(row["Assists"])), ("경고", int(row["Yellow Cards"])), ("퇴장", int(row["Red Cards"])),
    ]
    n = len(icon_stats)
    for i, (label, val) in enumerate(icon_stats):
        x = (i + 0.5) / n
        ax_icons.text(x, 0.75, label, ha="center", fontsize=8, color="#666")
        ax_icons.text(x, 0.2, str(val), ha="center", fontsize=13, fontweight="bold")

    # ── SCOUTING REPORT 텍스트 (템플릿, 실제 랭킹 숫자 근거) ──
    ax_text = fig.add_axes([0.06, 0.60, 0.88, 0.20])
    ax_text.axis("off")
    ax_text.text(0, 1.0, "SCOUTING REPORT", fontsize=13, fontweight="bold", color=color, va="top")
    report_text = _top_strengths_text(row, rankings)
    ax_text.text(0, 0.82, report_text, fontsize=9.5, va="top", wrap=True, linespacing=1.6)

    # ── 포지션 내 랭킹 표 (2행 x 5열, 약어 밑에 한국어 지표명 표기) ──
    ax_rank = fig.add_axes([0.06, 0.27, 0.88, 0.29])
    ax_rank.set_xlim(0, 5); ax_rank.set_ylim(0, 2); ax_rank.axis("off")
    ax_rank.text(0, 2.15, f"포지션({position}) 내 랭킹 비교대상 {n_peers}명", fontsize=10, fontweight="bold")
    for i, r in enumerate(rankings[:10]):
        col_i, row_i = i % 5, 1 - i // 5
        cx = col_i + 0.5
        cy = row_i + 0.55
        ax_rank.text(cx, cy + 0.32, r["abbr"], ha="center", fontsize=10, fontweight="bold")
        ax_rank.text(cx, cy - 0.02, f"{r['rank']}위", ha="center", fontsize=9, color=color, fontweight="bold")
        # 퍼센타일 바
        bar_w = 0.7
        ax_rank.add_patch(patches.Rectangle((cx - bar_w / 2, cy - 0.22), bar_w, 0.08,
                                             facecolor="#e0e0e0", edgecolor="none"))
        ax_rank.add_patch(patches.Rectangle((cx - bar_w / 2, cy - 0.22), bar_w * r["percentile"] / 100, 0.08,
                                             facecolor=color, edgecolor="none"))
        # 약어 뜻풀이 (한국어) — 일반 사용자는 TK/IC 같은 약어를 못 알아봐서 추가
        kr_label = KOREAN_STAT_NAME.get(r["column"], r["column"])
        ax_rank.text(cx, cy - 0.38, kr_label, ha="center", fontsize=6.3, color="#666")

    # ── 포지션 미니 필드 마커 ──
    ax_pitch = fig.add_axes([0.30, 0.05, 0.40, 0.19])
    ax_pitch.set_xlim(0, 1); ax_pitch.set_ylim(0, 1)
    ax_pitch.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#999", linewidth=1))
    ax_pitch.axhline(0.5, color="#ccc", linewidth=0.5)
    px, py = PITCH_POSITION_XY.get(position, (0.5, 0.5))
    ax_pitch.scatter([px], [py], s=220, color=color, zorder=3)
    ax_pitch.set_xticks([]); ax_pitch.set_yticks([])
    ax_pitch.set_title("포지션", fontsize=9, color="#666")

    footer = ("데이터: 프리미어리그 2024-25 시즌 스탯 (좌표 데이터 없어 실제 히트맵 대신 "
              "포지션 동료 대비 퍼센타일 랭킹으로 표기)")
    fig.text(0.5, 0.015, footer, ha="center", fontsize=6.5, color="#999")

    output_path = output_path or f"output/scouting_report_{row['Player Name'].replace(' ', '_')}.png"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    for name in ["Mohamed Salah", "Erling Haaland", "Virgil van Dijk"]:
        try:
            path = render_scouting_card(name)
            print(f"{name} -> {path}")
        except ValueError as e:
            print(e)
