"""
AIマラソンコーチ - Streamlit App
ジャック・ダニエルズのVDOT理論に基づくトレーニング計画生成

Version: β0.9
"""

import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import re
from datetime import datetime, timedelta
import io

# =============================================
# アプリ設定
# =============================================
APP_NAME = "AIマラソンコーチ"
APP_VERSION = "β0.22"

# =============================================
# ページ設定
# =============================================
st.set_page_config(
    page_title=f"{APP_NAME} v{APP_VERSION}",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================
# カスタムCSS
# =============================================
st.markdown("""
<style>
    /* サイドバーを非表示 */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .version-tag {
        font-size: 0.9rem;
        color: #888;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .form-section-title {
        font-size: 1.1rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .vdot-display {
        background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .vdot-explanation {
        background-color: #E3F2FD;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        color: #1a1a1a;
    }
    .vdot-explanation h4 {
        color: #1565C0;
    }
    .vdot-explanation p, .vdot-explanation li {
        color: #333;
    }
    .warning-box {
        background-color: #FFF3E0;
        border-left: 4px solid #FF9800;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        color: #1a1a1a;
    }
    .warning-box h4 {
        color: #E65100;
    }
    .warning-box p {
        color: #333;
    }
    .success-box {
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        color: #1a1a1a;
    }
    .success-box h4 {
        color: #2E7D32;
    }
    .success-box p {
        color: #333;
    }
    .error-box {
        background-color: #FFEBEE;
        border-left: 4px solid #F44336;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        color: #1a1a1a;
    }
    .phase-explanation {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #1a1a1a;
    }
    .phase-explanation h4 {
        color: #1565C0;
    }
    .phase-explanation table {
        color: #333;
    }
    .phase-explanation th {
        color: #1565C0;
    }
    .phase-explanation td {
        color: #333;
    }
</style>
""", unsafe_allow_html=True)


# =============================================
# CSVデータの読み込みとキャッシュ
# =============================================
@st.cache_data
def load_csv_data():
    """CSVファイルを読み込み、検証ログを生成"""
    verification_log = {
        "success": False,
        "files": [],
        "vdot_range": {"min": None, "max": None},
        "columns": {},
        "errors": []
    }
    
    try:
        df_vdot_list = pd.read_csv("data/vdot_list.csv")
        verification_log["files"].append("vdot_list.csv")
        verification_log["columns"]["VDOT_list"] = list(df_vdot_list.columns)

        df_pace = pd.read_csv("data/vdot_pace.csv")
        verification_log["files"].append("vdot_pace.csv")

        df_pace.columns = df_pace.columns.str.strip()
        verification_log["columns"]["VDOT_pace"] = list(df_pace.columns)
        
        vdot_col = "VDot" if "VDot" in df_pace.columns else "VDOT"
        vdot_min = int(df_pace[vdot_col].min())
        vdot_max = int(df_pace[vdot_col].max())
        verification_log["vdot_range"]["min"] = vdot_min
        verification_log["vdot_range"]["max"] = vdot_max
        
        verification_log["success"] = True
        
        return df_vdot_list, df_pace, verification_log
        
    except FileNotFoundError as e:
        verification_log["errors"].append(f"ファイルが見つかりません: {str(e)}")
        return None, None, verification_log
    except Exception as e:
        verification_log["errors"].append(f"読み込みエラー: {str(e)}")
        return None, None, verification_log


# =============================================
# VDOT計算関数群
# =============================================
def time_to_seconds(time_str: str) -> int:
    """時間文字列を秒に変換"""
    if pd.isna(time_str):
        return None
    
    time_str = str(time_str).strip()
    
    if time_str.count(':') == 2:
        parts = time_str.split(':')
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + int(s)
    
    elif time_str.count(':') == 1:
        parts = time_str.split(':')
        if len(parts) == 2:
            m, s = parts
            if ':' in str(s):
                return int(m) * 60 + int(s.split(':')[0])
            return int(m) * 60 + int(s)
    
    try:
        return int(float(time_str))
    except:
        return None


def seconds_to_time(seconds: int, include_hours: bool = False) -> str:
    """秒を時間文字列に変換"""
    if seconds is None:
        return "N/A"
    
    seconds = round(seconds)
    
    if include_hours or seconds >= 3600:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}:{m:02d}:{s:02d}"
    else:
        m = seconds // 60
        s = seconds % 60
        return f"{m}:{s:02d}"


def calculate_vdot_from_time(df_vdot: pd.DataFrame, distance: str, time_seconds: int) -> dict:
    """タイムからVDOTを線型補完で算出"""
    result = {
        "vdot": None,
        "calculation_log": "",
        "reference_data": {}
    }
    
    distance_mapping = {
        "5km": "5000m",
        "5000m": "5000m",
        "10km": "10000m",
        "10000m": "10000m",
        "ハーフ": "HalfMarathon",
        "ハーフマラソン": "HalfMarathon",
        "half": "HalfMarathon",
        "フル": "Marathon",
        "フルマラソン": "Marathon",
        "marathon": "Marathon",
        "マラソン": "Marathon"
    }
    
    col_name = distance_mapping.get(distance, distance)
    
    if col_name not in df_vdot.columns:
        result["calculation_log"] = f"エラー: 距離 '{distance}' が見つかりません"
        return result
    
    vdot_times = []
    for _, row in df_vdot.iterrows():
        vdot = int(row['VDOT'])
        time_val = row[col_name]
        time_sec = time_to_seconds(str(time_val))
        if time_sec:
            vdot_times.append((vdot, time_sec))
    
    vdot_times.sort(key=lambda x: x[1], reverse=True)
    
    lower_vdot = None
    upper_vdot = None
    
    for i, (vdot, time_sec) in enumerate(vdot_times):
        if time_sec <= time_seconds:
            lower_vdot = (vdot, time_sec)
            if i > 0:
                upper_vdot = vdot_times[i - 1]
            break
    
    if lower_vdot is None:
        lower_vdot = vdot_times[-1]
        upper_vdot = vdot_times[-2] if len(vdot_times) > 1 else None
    
    if upper_vdot is None:
        result["vdot"] = float(lower_vdot[0])
        result["calculation_log"] = f"VDOT {lower_vdot[0]} を使用（範囲外のため最も近い値）"
        return result
    
    vdot_low, time_low = upper_vdot
    vdot_high, time_high = lower_vdot
    
    if vdot_low > vdot_high:
        vdot_low, time_low, vdot_high, time_high = vdot_high, time_high, vdot_low, time_low
    
    if time_low != time_high:
        ratio = (time_low - time_seconds) / (time_low - time_high)
        calculated_vdot = vdot_low + (vdot_high - vdot_low) * ratio
    else:
        calculated_vdot = vdot_low
    
    result["vdot"] = round(calculated_vdot, 2)
    result["reference_data"] = {
        "vdot_low": vdot_low,
        "time_low": seconds_to_time(time_low, True),
        "time_low_sec": time_low,
        "vdot_high": vdot_high,
        "time_high": seconds_to_time(time_high, True),
        "time_high_sec": time_high
    }
    result["calculation_log"] = (
        f"【計算過程】\n"
        f"参照データ: VDOT {vdot_low} = {seconds_to_time(time_low, True)}, "
        f"VDOT {vdot_high} = {seconds_to_time(time_high, True)}\n"
        f"入力タイム: {seconds_to_time(time_seconds, True)}\n"
        f"計算式: {vdot_low} + ({vdot_high} - {vdot_low}) × "
        f"({time_low} - {time_seconds}) / ({time_low} - {time_high})\n"
        f"= {vdot_low} + {vdot_high - vdot_low} × {ratio:.4f}\n"
        f"= {calculated_vdot:.2f}"
    )
    
    return result


def calculate_training_paces(df_pace: pd.DataFrame, vdot: float) -> dict:
    """VDOTから練習ペースを線型補完で算出"""
    result = {
        "paces": {},
        "calculation_log": "",
        "success": False
    }
    
    vdot_col = "VDot" if "VDot" in df_pace.columns else "VDOT"
    
    vdot_low = int(vdot)
    vdot_high = vdot_low + 1
    decimal_ratio = vdot - vdot_low
    
    row_low = df_pace[df_pace[vdot_col] == vdot_low]
    row_high = df_pace[df_pace[vdot_col] == vdot_high]
    
    if row_low.empty:
        result["calculation_log"] = f"エラー: VDOT {vdot_low} がファイルに存在しません"
        return result
    
    if row_high.empty:
        row_high = row_low
        decimal_ratio = 0
    
    row_low = row_low.iloc[0]
    row_high = row_high.iloc[0]
    
    pace_types = ["E_min", "E_max", "M", "T", "I", "R"]
    calculation_details = []
    
    for pace_type in pace_types:
        if pace_type not in df_pace.columns:
            continue
        
        pace_low_str = str(row_low[pace_type])
        pace_high_str = str(row_high[pace_type])
        
        pace_low_sec = time_to_seconds(pace_low_str)
        pace_high_sec = time_to_seconds(pace_high_str)
        
        if pace_low_sec is None or pace_high_sec is None:
            continue
        
        pace_sec = pace_low_sec + (pace_high_sec - pace_low_sec) * decimal_ratio
        pace_sec = round(pace_sec)
        
        result["paces"][pace_type] = {
            "seconds": pace_sec,
            "display": seconds_to_time(pace_sec)
        }
        
        calculation_details.append(
            f"  {pace_type}: {pace_low_sec}秒 + ({pace_high_sec}秒 - {pace_low_sec}秒) × {decimal_ratio:.2f} "
            f"= {pace_sec}秒 → {seconds_to_time(pace_sec)}/km"
        )
    
    if "E_min" in result["paces"] and "E_max" in result["paces"]:
        result["paces"]["E"] = {
            "display": f"{result['paces']['E_min']['display']}〜{result['paces']['E_max']['display']}",
            "min": result["paces"]["E_min"],
            "max": result["paces"]["E_max"]
        }
    
    result["calculation_log"] = (
        f"【練習ペース計算過程】\n"
        f"設定VDOT: {vdot} (VDOT {vdot_low} と VDOT {vdot_high} の間、比率 {decimal_ratio:.2f})\n"
        f"参照データ（VDOT {vdot_low}）: E={row_low['E_min']}〜{row_low['E_max']}, "
        f"M={row_low['M']}, T={row_low['T']}, I={row_low['I']}, R={row_low['R']}\n"
        f"参照データ（VDOT {vdot_high}）: E={row_high['E_min']}〜{row_high['E_max']}, "
        f"M={row_high['M']}, T={row_high['T']}, I={row_high['I']}, R={row_high['R']}\n"
        f"計算詳細:\n" + "\n".join(calculation_details)
    )
    
    result["success"] = True
    return result


def calculate_phase_vdots(current_vdot: float, target_vdot: float, num_phases: int = 4) -> list:
    """フェーズごとのVDOT目標を計算（4フェーズ固定）
    
    フェーズ1: 現在のVDOT（基礎構築期）
    フェーズ2〜4: 段階的に目標VDOTに近づける
    """
    vdot_diff = target_vdot - current_vdot
    # フェーズ1は現在VDOT、残り3フェーズで目標に到達
    step = vdot_diff / (num_phases - 1) if num_phases > 1 else vdot_diff
    
    phase_vdots = []
    for i in range(num_phases):
        if i == 0:
            # フェーズ1は現在のVDOT
            phase_vdots.append(round(current_vdot, 2))
        else:
            # フェーズ2以降は段階的に上昇
            phase_vdot = round(current_vdot + step * i, 2)
            phase_vdots.append(phase_vdot)
    
    return phase_vdots


def calculate_marathon_time_from_vdot(df_vdot: pd.DataFrame, vdot: float) -> str:
    """VDOTからマラソンタイムを線形補間で計算"""
    try:
        vdot_low = int(vdot)
        vdot_high = vdot_low + 1
        decimal_ratio = vdot - vdot_low
        
        row_low = df_vdot[df_vdot['VDOT'] == vdot_low]
        row_high = df_vdot[df_vdot['VDOT'] == vdot_high]
        
        if row_low.empty:
            return "N/A"
        
        time_low_str = str(row_low.iloc[0]['Marathon'])
        time_low_sec = time_to_seconds(time_low_str)
        
        if row_high.empty or time_low_sec is None:
            return time_low_str if time_low_sec else "N/A"
        
        time_high_str = str(row_high.iloc[0]['Marathon'])
        time_high_sec = time_to_seconds(time_high_str)
        
        if time_high_sec is None:
            return time_low_str
        
        # 線形補間（VDOTが上がるとタイムは短くなる）
        time_sec = time_low_sec + (time_high_sec - time_low_sec) * decimal_ratio
        
        # HH:MM:SS形式に変換
        hours = int(time_sec // 3600)
        minutes = int((time_sec % 3600) // 60)
        seconds = int(time_sec % 60)
        
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    except Exception:
        return "N/A"


def get_training_start_date(race_date: datetime, min_weeks: int = 12) -> datetime:
    """トレーニング開始日を計算（最低12週確保、月曜始まり）"""
    today = datetime.now()
    
    # レースまでの週数を計算
    days_until_race = (race_date - today).days
    weeks_until_race = days_until_race // 7
    
    # 12週未満の場合は過去に遡る
    if weeks_until_race < min_weeks:
        # レース日から12週前の月曜日を計算
        start_date = race_date - timedelta(weeks=min_weeks)
    else:
        # 今日から始める
        start_date = today
    
    # 月曜日に調整（次の月曜日）
    days_until_monday = (7 - start_date.weekday()) % 7
    if days_until_monday == 0 and start_date.weekday() != 0:
        days_until_monday = 7
    start_date = start_date + timedelta(days=days_until_monday)
    
    # もし開始日が月曜でない場合、前の月曜に調整
    if start_date.weekday() != 0:
        start_date = start_date - timedelta(days=start_date.weekday())
    
    return start_date


# =============================================
# Gemini API 設定
# =============================================
def get_gemini_model():
    """Gemini APIモデルを取得"""
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config={
            "temperature": 0.7,
            "top_p": 0.95,
            "max_output_tokens": 16384,
        }
    )
    return model


def create_training_prompt(user_data: dict, vdot_info: dict, pace_info: dict, target_vdot_info: dict, df_pace: pd.DataFrame, training_weeks: int, start_date: datetime, df_vdot: pd.DataFrame = None) -> str:
    """トレーニング計画生成用のプロンプトを作成"""
    
    paces = pace_info.get("paces", {}) if pace_info else {}
    current_vdot = vdot_info['vdot']
    target_vdot = target_vdot_info['vdot'] if target_vdot_info else current_vdot
    vdot_diff = round(target_vdot - current_vdot, 2)
    
    # 元の目標VDOTと調整済み目標VDOTの情報
    original_target_vdot = user_data.get("original_target_vdot")
    adjusted_target_vdot = user_data.get("adjusted_target_vdot")
    
    # 中間目標マラソンタイムを計算
    adjusted_marathon_time = ""
    if adjusted_target_vdot and df_vdot is not None:
        adjusted_marathon_time = calculate_marathon_time_from_vdot(df_vdot, adjusted_target_vdot)
    
    # 過去遡り開始の判定
    today = datetime.now()
    is_past_start = start_date < today
    
    # VDOT調整の説明文（プロンプト用の情報提供のみ、Geminiへの出力指示は出力テンプレートで行う）
    vdot_adjustment_note = ""
    if adjusted_target_vdot and original_target_vdot and adjusted_target_vdot != original_target_vdot:
        vdot_adjustment_note = f"""
## ⚠️ 目標VDOTの調整について（情報）

ユーザーが入力した目標タイム（{user_data.get('target_time', '')}、VDOT {original_target_vdot}）と現在のVDOT（{current_vdot}）の差が3.0を超えています。
今回のトレーニング計画では中間目標を設定しています：

- 中間目標VDOT: {adjusted_target_vdot}（VDOT差 3.0）
- 中間目標マラソンタイム: {adjusted_marathon_time}
- 最終目標: VDOT {original_target_vdot} / {user_data.get('target_time', '')}

※この情報は出力テンプレートの「基本情報」セクションに既に反映されています。追加の説明セクションを作成しないでください。
"""
    
    # 過去遡り開始の説明（出力テンプレートに直接反映する）
    past_start_note = ""
    if is_past_start:
        past_start_note = f"""
※ 本プログラムは最低12週間のトレーニング期間を確保します。レース日までの期間が12週間に満たないため、開始日を{start_date.strftime('%Y/%m/%d')}（過去の日付）に設定しています。実際には本日から計画を参考にしてください。
"""
    
    # レース日（フォーマット統一: YYYY/MM/DD）
    race_date_raw = user_data.get("race_date", "")
    try:
        race_dt = datetime.strptime(race_date_raw, "%Y-%m-%d")
        race_date_str = race_dt.strftime("%Y/%m/%d")
        race_weekday = ["月", "火", "水", "木", "金", "土", "日"][race_dt.weekday()]
        race_date_with_day = f"{race_dt.strftime('%m/%d')}（{race_weekday}）"
    except:
        race_date_str = race_date_raw
        race_date_with_day = race_date_raw
    
    # フェーズは4つ固定
    num_phases = 4
    weeks_per_phase = training_weeks // num_phases
    remaining_weeks = training_weeks % num_phases
    
    # フェーズごとのVDOT目標を計算
    phase_vdots = calculate_phase_vdots(current_vdot, target_vdot, num_phases)
    
    # 各フェーズのペース情報を生成
    phase_paces_info = []
    for i, phase_vdot in enumerate(phase_vdots):
        phase_pace = calculate_training_paces(df_pace, phase_vdot)
        phase_paces = phase_pace.get("paces", {})
        phase_paces_info.append({
            "phase": i + 1,
            "vdot": phase_vdot,
            "E": phase_paces.get('E', {}).get('display', 'N/A'),
            "M": phase_paces.get('M', {}).get('display', 'N/A'),
            "T": phase_paces.get('T', {}).get('display', 'N/A'),
            "I": phase_paces.get('I', {}).get('display', 'N/A'),
            "R": phase_paces.get('R', {}).get('display', 'N/A'),
        })
    
    # フェーズ情報をテキスト化
    phase_info_text = ""
    for p in phase_paces_info:
        phase_info_text += f"""
### フェーズ{p['phase']}（VDOT {p['vdot']}）
| ペース | 設定 |
|:---|:---|
| E (Easy) | {p['E']}/km |
| M (Marathon) | {p['M']}/km |
| T (Threshold) | {p['T']}/km |
| I (Interval) | {p['I']}/km |
| R (Repetition) | {p['R']}/km |
"""
    
    # 練習レースをQトレーニングとして処理
    practice_races_note = ""
    if user_data.get('practice_races'):
        practice_races_note = f"""
## ⚠️ 練習レースについて（最重要 - 日付厳守）

以下の練習レースは**入力された日付に正確に配置**してください。日付を変更してはいけません：

```
{user_data.get('practice_races')}
```

**練習レースの目的設定（重要）：**
練習レースは本番レースのための「練習」です。全ての練習レースを全力で走るのではなく、以下のようにバリエーションを持たせてください：

- **ハーフマラソン（序盤）**: Mペース〜Tペースで余裕を持って走る（本番ペースの確認）
- **ハーフマラソン（中盤〜終盤）**: 本番想定ペースで走り、レース感覚を養う
- **フルマラソン（練習レース）**: 以下のいずれかを提案
  - 30kmまでMペースで走り、残り12kmはEペースでジョグ
  - 全体をMペースより10〜15秒/km遅いペースで走り通す
  - 前半抑えめ、後半ビルドアップ

**練習レースのルール：**
1. **日付は絶対に変更しないこと**（入力された日付をそのまま使用）
2. 練習レースはQトレーニング（ポイント練習）の1回としてカウント
3. 練習レースがある週は、他のポイント練習を調整して週{user_data.get('point_training_days', '2')}回に収める
4. **レース前日は軽いジョグまたは完全休養**（ポイント練習禁止）
5. **レース前々日もポイント練習禁止**（Eペースジョグのみ）
6. レース翌日はリカバリージョグ
"""
    
    # 開始日のフォーマット
    start_date_str = start_date.strftime("%Y/%m/%d")
    
    prompt = f"""# Role
あなたは、ジャック・ダニエルズ博士の直接指導を受け、「ランニング・フォーミュラ」を20年以上実践してきたマラソン専属コーチです。

【あなたの目的】
ユーザーの年齢・性別・走力・生活環境・過去の怪我などを深く理解し、目標タイムを達成するための「実現可能で安全な」トレーニング計画を提案すること。

【あなたの指導哲学】
- 「Train where you are」：今の走力で練習し、目標の走力でレースに臨む
- 「ピラミッド型トレーニング」：E（低強度）を土台とし、M/T（中強度）で支え、I/R（高強度）は頂点にわずかに置く
- 「怪我なくして成長なし、ではない」：オーバートレーニングは最大の敵。選手生命を守ることが最優先

あなたは温かく親しみやすい「優しい先生」ですが、トレーニングの質については一切妥協しません。科学的根拠のない練習は決して提案せず、すべてのメニューにダニエルズ理論に基づく明確な目的があります。

あなたは以下の専門的手法を熟知し、ユーザーの状況に応じて適切に使い分けます：
ダニエルズ理論（VDOT）、ピラミッド型モデル、土日セット練習、ダブルスレッショルド、2部練

# Tone & Style
- **語り口:** 常に温かく、親しみやすく、ポジティブな「優しい先生」。ユーザーを褒めて伸ばすスタイル。
- **専門性:** ダニエルズ理論（E, M, T, I, Rペース）を正確に使用し、本格的な指導を行う。
- **配慮:** オーバートレーニングを最も嫌います。ユーザーの生活背景を考慮し、現実的なメニューを提案します。

# 専門用語辞書（コーチが熟知し、適切に運用する知識）

1. **ダニエルズ理論（VDOT）**
定義：ジャック・ダニエルズ博士による走力指標。現在の走力からE/M/T/I/Rの5つの最適ペースを算出する。
運用：すべてのトレーニング計画の基盤。ペース設定は必ずVDOTに基づく。

2. **ピラミッド型モデル（Pyramidal Training）**
定義：低強度(E)を底辺(70-80%)、中強度(M/T)を中段(15-20%)、高強度(I/R)を頂点(5-10%)とする強度配分。
運用：週間メニューを組む際、各フェーズの目的を踏まえつつ基本的にこの比率を守る。

3. **土日セット練習（セット練）**
定義：1日目に高強度でグリコーゲンを枯渇させ、2日目に疲労状態で距離走を行う手法。
運用：ユーザーが希望した場合「かつ」実施可能な週のみ導入。ダニエルズ理論・ピラミッド型モデルを優先する。

4. **ダブルスレッショルド**
定義：1日に午前・午後の2回、閾値強度の練習を行うノルウェー方式。
運用：ユーザーが希望した場合「かつ」実施可能な週のみ導入。ダニエルズ理論・ピラミッド型モデルを優先する。

5. **2部練（午前午後分割）**
定義：1回の走行による筋損傷を抑えつつ、1日の総負荷を高める手法。
運用：ユーザーが希望した場合「かつ」実施可能な週のみ導入。ダニエルズ理論・ピラミッド型モデルを優先する。

# ユーザー情報
- ニックネーム: {user_data.get('name', '不明')}
- 年齢: {user_data.get('age', '不明')}歳
- 性別: {user_data.get('gender', '不明')}
- 現在のベストタイム: {user_data.get('current_time', '不明')}（フルマラソン）
- 目標タイム: {user_data.get('target_time', '不明')}
- 本番レース: {user_data.get('race_name', '不明')}（{race_date_str} {race_weekday}曜日）
- トレーニング期間: {training_weeks}週間
- トレーニング開始日: {start_date_str}（月曜日）
- 練習レース: {user_data.get('practice_races', 'なし')}
- 週間走行距離: {user_data.get('weekly_distance', '不明')}km
- 練習可能日数: {user_data.get('training_days', '不明')}日/週
- ポイント練習可能回数: {user_data.get('point_training_days', '不明')}回/週

# 🚨 最優先事項（AIコーチへの連絡事項）
以下はユーザーからの特別な要望・制約です。**必ず計画に反映してください。無視してはいけません。**

```
{user_data.get('concerns', 'なし')}
```

※上記の内容がある場合、怪我の配慮、特定日の練習不可、練習スタイルの希望など、**全てのルールより優先**して対応すること。

# VDOT情報
- 現在のVDOT: {current_vdot}
- 目標VDOT: {target_vdot}
- VDOT差: {vdot_diff}

{vdot_adjustment_note}

# 4フェーズ構成（各フェーズ約{weeks_per_phase}週間）

## フェーズごとのVDOTとペース設定
{phase_info_text}

{practice_races_note}

# 出力指示
以下の形式で、レースまでの全トレーニング計画を出力してください。

## 出力形式（Markdown）

# 🏃‍♂️ {user_data.get('name', 'ユーザー')}さんのトレーニング計画

## 📊 基本情報
- 現在のVDOT: {current_vdot} → 今回の目標VDOT: {target_vdot}（VDOT差: {vdot_diff}）
- 今回の目標タイム: {adjusted_marathon_time if adjusted_marathon_time else user_data.get('target_time', '')}（{user_data.get('race_name', '')} {race_date_str}）
- 最終目標タイム: {user_data.get('target_time', '')}（VDOT {original_target_vdot if original_target_vdot else target_vdot}）※中間目標を達成後、次のサイクルで目指す
- トレーニング期間: {training_weeks}週間（4フェーズ）
- 開始日: {start_date_str}（月曜日）
{past_start_note}
（※中間目標を設定していない場合は「今回の目標タイム」と「最終目標タイム」は同じになります。その場合は「最終目標タイム」の行は出力しないでください。）
（※past_start_noteが空の場合は、その行は出力しないでください。）

---

## 📖 VDOTとペースについて

### VDOTとは
VDOTは、ジャック・ダニエルズ博士が考案した走力指標です。現在のタイムから算出され、適切なトレーニングペースを導き出すことができます。

### 各ペースの意味
| ペース | 目的 | 説明 |
|:---|:---|:---|
| E (Easy) | 有酸素能力の向上 | 会話ができる楽なペース。全体の走行距離の大部分をこのペースで走ります。 |
| M (Marathon) | レースペース適応 | フルマラソンの目標ペース。レース感覚を養います。 |
| T (Threshold) | 乳酸閾値の向上 | 「快適にきつい」ペース。20〜30分維持できる強度です。 |
| I (Interval) | 最大酸素摂取量向上 | 3〜5分維持できるきついペース。インターバル走で使用します。 |
| R (Repetition) | スピード・効率向上 | 最大スピードに近いペース。短い距離の反復で使用します。 |

---

## 📈 4フェーズ構成

| フェーズ | 期間 | 目標VDOT | 主な目的 |
|:---|:---|:---|:---|
| フェーズ1（基礎構築期） | 第1〜{weeks_per_phase}週 | {phase_vdots[0]} | 基礎体力の構築、怪我からの回復 |
| フェーズ2（強化期） | 第{weeks_per_phase+1}〜{weeks_per_phase*2}週 | {phase_vdots[1]} | 持久力・スピードの強化 |
| フェーズ3（実践期） | 第{weeks_per_phase*2+1}〜{weeks_per_phase*3}週 | {phase_vdots[2]} | レースペースへの適応 |
| フェーズ4（調整期） | 第{weeks_per_phase*3+1}〜{training_weeks}週 | {phase_vdots[3]} | テーパリング、最終調整 |

### フェーズの説明

**フェーズ1（基礎構築期）**: 
現在のVDOT {current_vdot} に基づいたペースで、基礎的な有酸素能力を構築します。Eペースを中心に走り込み、身体をトレーニングに適応させます。

**フェーズ2（強化期）**: 
VDOT {phase_vdots[1]} にペースを上げ、Tペースやインターバルを増やして持久力とスピードを強化します。

**フェーズ3（実践期）**: 
VDOT {phase_vdots[2]} で、Mペースでのロング走やレースペースでの練習を増やし、本番に向けた実践的なトレーニングを行います。

**フェーズ4（調整期）**: 
目標VDOT {target_vdot} に到達。練習量を徐々に減らし（テーパリング）、疲労を抜いて本番に備えます。

---

## 📋 週間トレーニング計画

**重要**: 各週は**月曜日始まり〜日曜日終わり**で構成してください。

### フェーズ1（VDOT {phase_vdots[0]}）- 基礎構築期

**このフェーズのペース設定:**
| ペース | 設定 |
|:---|:---|
| E (Easy) | {phase_paces_info[0]['E']}/km |
| M (Marathon) | {phase_paces_info[0]['M']}/km |
| T (Threshold) | {phase_paces_info[0]['T']}/km |
| I (Interval) | {phase_paces_info[0]['I']}/km |
| R (Repetition) | {phase_paces_info[0]['R']}/km |

#### 第1週（MM/DD（月）〜MM/DD（日））

| 日付 | メニュー | 距離 | ペース | 先生からのアドバイス |
|:---|:---|:---|:---|:---|
| MM/DD（月） | ... | ... | ... | ... |
| MM/DD（火） | ... | ... | ... | ... |
| MM/DD（水） | ... | ... | ... | ... |
| MM/DD（木） | ... | ... | ... | ... |
| MM/DD（金） | ... | ... | ... | ... |
| MM/DD（土） | ... | ... | ... | ... |
| MM/DD（日） | ... | ... | ... | ... |

（以下、全週・全フェーズを出力）

---

## ⚠️ 注意事項
（全体を通しての注意点を5項目程度）

## 💪 コーチからのメッセージ
（励ましのメッセージ）

---
*Generated by {APP_NAME} v{APP_VERSION}*

# 重要な出力ルール（必ず守ること）

## 【最重要】週の構成ルール
1. **週は必ず月曜日始まり〜日曜日終わり**の7日間で構成すること
2. **全ての週で月〜日の7日間全てを出力すること**（レースウィークも含む）
3. **日付は連続していること**（例：1/31（土）の翌日は2/1（日）であること。曜日を正確に計算すること）
4. 日付は「MM/DD（曜日）」形式で記載。**曜日は必ず正確に計算すること**
5. 2026年のカレンダーを正確に参照し、曜日を間違えないこと
6. **各週の表の後に「週間走行距離: XXkm」を必ず記載すること**

## 【重要】表のフォーマット
7. **週間トレーニング表のヘッダーは「日付 | メニュー | 距離 | ペース | 先生からのアドバイス」とすること**（「ポイント」ではなく「先生からのアドバイス」）
8. **HTMLタグは絶対に使用禁止**：`<hr>`, `<h4>`, `<p>`, `<strong>`, `<br>`, `<div>`, `<span>`等、一切のHTMLタグを出力してはいけない。Markdownのみで記述すること
9. 表形式は省略・改変せず、指定されたフォーマットを厳守すること
10. **出力テンプレートに指定されていないセクションを勝手に追加しないこと**（「トレーニング開始日について」等の独自セクションを作成しない）

## 【重要】練習レースの扱い
11. **練習レースは入力された日付に正確に配置すること**（日付を変更してはいけない）
12. 練習レースはQトレーニング（ポイント練習）としてカウント
13. **練習レース前日・前々日はポイント練習禁止**（Eペースジョグまたは休養のみ）
14. 練習レース翌日はリカバリージョグ

## 【最重要】ポイント練習（Q）のルール
15. **ポイント練習は週{user_data.get('point_training_days', '2')}回**とすること（練習レースを含む）
16. **ポイント練習を2日連続で行ってはいけない**（必ず間に1日以上の回復日を入れる。ただし「土日セット練」が要望されている場合は例外）
17. ポイント練習の種類：インターバル走、テンポ走、ペース走、ロング走（25km以上）、練習レース
18. **ロング走（25km以上）はポイント練習（Q）としてカウント**すること
19. 練習レースがある週は、練習レースの前後に回復日を確保しつつ、週のポイント練習回数を調整すること

## トレーニング内容ルール
20. 各フェーズで、上記で指定したそのフェーズのVDOTに対応したペースを必ず使用すること
21. **フェーズ1は現在のVDOT（{current_vdot}）**で練習すること
22. 週間走行距離は{user_data.get('weekly_distance', '不明')}kmを目安にすること
23. トレーニング開始日は{start_date_str}（月曜日）から始めること
24. 全{training_weeks}週間分のメニューを出力すること
25. 最終週はレースウィークとしてテーパリングを入れ、レース当日まで7日間出力すること
26. レース日は必ず{race_date_with_day}とすること（この日付を最終日として逆算してスケジュールを組むこと）
"""
    
    return prompt


# =============================================
# MDダウンロード用関数
# =============================================
def create_md_download(content: str) -> bytes:
    """Markdownファイルをダウンロード用バイトに変換（UTF-8 BOM付き）"""
    bom = b'\xef\xbb\xbf'
    content_bytes = content.encode('utf-8')
    return bom + content_bytes


def sanitize_gemini_output(content: str) -> str:
    """Geminiの出力からHTMLタグを除去してMarkdownのみにする"""
    import re
    
    # HTMLタグを含む行を検出して除去
    lines = content.split('\n')
    cleaned_lines = []
    
    # HTMLタグのパターン
    html_patterns = [
        r'<hr[^>]*>',
        r'</?h[1-6][^>]*>',
        r'</?p[^>]*>',
        r'</?strong[^>]*>',
        r'</?em[^>]*>',
        r'</?div[^>]*>',
        r'</?span[^>]*>',
        r'</?br[^>]*>',
        r'</?ul[^>]*>',
        r'</?li[^>]*>',
        r'</?ol[^>]*>',
        r'</?a[^>]*>',
        r'</?table[^>]*>',
        r'</?tr[^>]*>',
        r'</?td[^>]*>',
        r'</?th[^>]*>',
    ]
    
    for line in lines:
        # 行にHTMLタグが含まれているかチェック
        has_html = False
        for pattern in html_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                has_html = True
                break
        
        # HTMLタグを含む行は除去
        if has_html:
            continue
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


# =============================================
# セッション状態の初期化
# =============================================
def init_session_state():
    """セッション状態を初期化"""
    if "form_submitted" not in st.session_state:
        st.session_state.form_submitted = False
    if "user_data" not in st.session_state:
        st.session_state.user_data = {}
    if "calculated_vdot" not in st.session_state:
        st.session_state.calculated_vdot = None
    if "target_vdot" not in st.session_state:
        st.session_state.target_vdot = None
    if "training_paces" not in st.session_state:
        st.session_state.training_paces = None
    if "training_plan" not in st.session_state:
        st.session_state.training_plan = None
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "training_weeks" not in st.session_state:
        st.session_state.training_weeks = 12
    if "start_date" not in st.session_state:
        st.session_state.start_date = None


# =============================================
# メイン UI
# =============================================
def main():
    init_session_state()
    
    # ヘッダー
    st.markdown(f'<h1 class="main-header">🏃 {APP_NAME}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="version-tag">Version {APP_VERSION}</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ジャック・ダニエルズのVDOT理論に基づく、あなただけのトレーニング計画</p>', unsafe_allow_html=True)
    
    # データ読み込み
    df_vdot, df_pace, verification_log = load_csv_data()
    
    if not verification_log["success"]:
        st.error("CSVデータの読み込みに失敗しました。")
        for error in verification_log["errors"]:
            st.error(error)
        return
    
    st.session_state.data_loaded = True
    st.session_state.df_vdot = df_vdot
    st.session_state.df_pace = df_pace
    
    # API Key確認
    if not st.secrets.get("GEMINI_API_KEY", ""):
        st.error("⚠️ Gemini API Keyが設定されていません。Streamlit CloudのSecretsで設定してください。")
        return
    
    # メインコンテンツ
    if not st.session_state.form_submitted:
        # ================== 入力フォーム ==================
        st.markdown("### 📝 あなたの情報を入力してください")
        
        with st.form("user_info_form"):
            # 基本情報
            st.markdown('<div class="form-section-title">👤 基本情報</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                name = st.text_input("ニックネーム", placeholder="例: 太郎")
            with col2:
                age = st.number_input("年齢", min_value=10, max_value=100, value=40)
            with col3:
                gender = st.selectbox("性別", ["男性", "女性", "その他"])
            
            st.markdown("---")
            
            # タイム情報
            st.markdown('<div class="form-section-title">⏱ タイム情報</div>', unsafe_allow_html=True)
            
            # ベストタイム（+/- ボタン形式）
            st.markdown("**現在のベストタイム（フルマラソン）**")
            col1, col2, col3 = st.columns(3)
            with col1:
                current_h = st.number_input("時間", min_value=2, max_value=6, value=3, step=1, key="current_h")
            with col2:
                current_m = st.number_input("分", min_value=0, max_value=59, value=30, step=1, key="current_m")
            with col3:
                current_s = st.number_input("秒", min_value=0, max_value=59, value=0, step=1, key="current_s")
            
            # 目標タイム（+/- ボタン形式）
            st.markdown("**目標タイム（フルマラソン）**")
            col1, col2, col3 = st.columns(3)
            with col1:
                target_h = st.number_input("時間", min_value=2, max_value=6, value=3, step=1, key="target_h")
            with col2:
                target_m = st.number_input("分", min_value=0, max_value=59, value=15, step=1, key="target_m")
            with col3:
                target_s = st.number_input("秒", min_value=0, max_value=59, value=0, step=1, key="target_s")
            
            st.markdown("---")
            
            # レース情報
            st.markdown('<div class="form-section-title">🏁 レース情報</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                race_name = st.text_input("本番レース名", placeholder="例: 東京マラソン")
                race_date = st.date_input("本番レース日", value=datetime.now() + timedelta(days=90))
            with col2:
                practice_races = st.text_area("練習レース（任意）", placeholder="例: 1/11 NYハーフ\n1/18 赤羽ハーフ", height=100)
            
            st.markdown("---")
            
            # 練習情報
            st.markdown('<div class="form-section-title">🏃‍♂️ 練習情報</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                weekly_distance = st.text_input("週間走行距離（km）", placeholder="例: 50-60")
            with col2:
                training_days = st.selectbox("練習可能日数/週", [3, 4, 5, 6, 7], index=3)
            with col3:
                point_training_days = st.selectbox("ポイント練習回数/週", [1, 2, 3], index=1)
            
            concerns = st.text_area(
                "AIコーチへの連絡事項（任意）", 
                placeholder="例: 右膝に違和感がある、2/5は練習できない、土日セット練希望",
                height=80
            )
            
            st.markdown("---")
            
            # 送信ボタン
            submitted = st.form_submit_button("🚀 トレーニング計画を作成", use_container_width=True, type="primary")
            
            if submitted:
                # バリデーション
                errors = []
                if not name:
                    errors.append("ニックネームを入力してください")
                if not race_name:
                    errors.append("本番レース名を入力してください")
                
                # タイムを秒に変換
                current_seconds = current_h * 3600 + current_m * 60 + current_s
                target_seconds = target_h * 3600 + target_m * 60 + target_s
                
                # タイム文字列を生成
                current_time = f"{current_h}:{current_m:02d}:{current_s:02d}"
                target_time = f"{target_h}:{target_m:02d}:{target_s:02d}"
                
                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    # VDOT計算
                    vdot_result = calculate_vdot_from_time(df_vdot, "フルマラソン", current_seconds)
                    target_vdot_result = calculate_vdot_from_time(df_vdot, "フルマラソン", target_seconds)
                    
                    # VDOT差チェック
                    if vdot_result["vdot"] and target_vdot_result["vdot"]:
                        vdot_diff = target_vdot_result["vdot"] - vdot_result["vdot"]
                        
                        # VDOT差が3.0を超える場合、調整済み目標VDOTを計算
                        original_target_vdot = target_vdot_result["vdot"]
                        adjusted_target_vdot = None
                        if vdot_diff > 3.0:
                            # 現在VDOTから3.0上げた値を調整済み目標とする
                            adjusted_target_vdot = round(vdot_result["vdot"] + 3.0, 2)
                        
                        # データ保存
                        st.session_state.user_data = {
                            "name": name,
                            "age": age,
                            "gender": gender,
                            "current_time": current_time,
                            "target_time": target_time,
                            "race_name": race_name,
                            "race_date": race_date.strftime("%Y-%m-%d"),
                            "practice_races": practice_races,
                            "weekly_distance": weekly_distance,
                            "training_days": training_days,
                            "point_training_days": point_training_days,
                            "concerns": concerns,
                            "vdot_diff": round(vdot_diff, 2),
                            "original_target_vdot": original_target_vdot,
                            "adjusted_target_vdot": adjusted_target_vdot
                        }
                        
                        st.session_state.calculated_vdot = vdot_result
                        st.session_state.target_vdot = target_vdot_result
                        
                        if vdot_result["vdot"]:
                            pace_result = calculate_training_paces(df_pace, vdot_result["vdot"])
                            st.session_state.training_paces = pace_result
                        
                        # トレーニング期間の計算（最低12週）
                        race_dt = datetime.combine(race_date, datetime.min.time())
                        start_date = get_training_start_date(race_dt, min_weeks=12)
                        training_weeks = max(12, ((race_dt - start_date).days // 7) + 1)
                        
                        st.session_state.training_weeks = training_weeks
                        st.session_state.start_date = start_date
                        
                        st.session_state.form_submitted = True
                        st.rerun()
    
    else:
        # ================== 結果表示 ==================
        user_data = st.session_state.user_data
        vdot_info = st.session_state.calculated_vdot
        pace_info = st.session_state.training_paces
        target_vdot = st.session_state.target_vdot
        paces = pace_info.get("paces", {}) if pace_info else {}
        vdot_diff = user_data.get("vdot_diff", 0)
        training_weeks = st.session_state.training_weeks
        start_date = st.session_state.start_date
        
        # VDOT情報表示
        target_vdot_display = ""
        if target_vdot and target_vdot.get("vdot"):
            target_vdot_display = f'<span style="margin-left: 2rem;">🎯 目標VDOT: <strong>{target_vdot["vdot"]}</strong></span>'
        
        st.markdown(f"""
<div class="vdot-display">
    <h3 style="margin: 0 0 1rem 0; color: white;">📊 {user_data.get('name', '')}さんのVDOT計算結果</h3>
    <div style="font-size: 1.3rem; margin-bottom: 1rem;">
        🏃 現在のVDOT: <strong>{vdot_info['vdot']}</strong>{target_vdot_display}
        <span style="margin-left: 2rem;">📈 VDOT差: <strong>{vdot_diff}</strong></span>
    </div>
    <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.5rem; text-align: center;">
        <div style="background: rgba(255,255,255,0.2); padding: 0.5rem; border-radius: 8px;">
            <div style="font-size: 0.8rem;">E (Easy)</div>
            <div style="font-weight: bold;">{paces.get('E', {}).get('display', 'N/A')}/km</div>
        </div>
        <div style="background: rgba(255,255,255,0.2); padding: 0.5rem; border-radius: 8px;">
            <div style="font-size: 0.8rem;">M (Marathon)</div>
            <div style="font-weight: bold;">{paces.get('M', {}).get('display', 'N/A')}/km</div>
        </div>
        <div style="background: rgba(255,255,255,0.2); padding: 0.5rem; border-radius: 8px;">
            <div style="font-size: 0.8rem;">T (Threshold)</div>
            <div style="font-weight: bold;">{paces.get('T', {}).get('display', 'N/A')}/km</div>
        </div>
        <div style="background: rgba(255,255,255,0.2); padding: 0.5rem; border-radius: 8px;">
            <div style="font-size: 0.8rem;">I (Interval)</div>
            <div style="font-weight: bold;">{paces.get('I', {}).get('display', 'N/A')}/km</div>
        </div>
        <div style="background: rgba(255,255,255,0.2); padding: 0.5rem; border-radius: 8px;">
            <div style="font-size: 0.8rem;">R (Repetition)</div>
            <div style="font-weight: bold;">{paces.get('R', {}).get('display', 'N/A')}/km</div>
        </div>
    </div>
</div>
        """, unsafe_allow_html=True)
        
        # 調整済み目標VDOTの取得
        adjusted_target_vdot = user_data.get("adjusted_target_vdot")
        original_target_vdot = user_data.get("original_target_vdot")
        
        # 実際に使用する目標VDOT（調整済みがあればそれを使用）
        effective_target_vdot = adjusted_target_vdot if adjusted_target_vdot else target_vdot['vdot']
        
        # 中間目標マラソンタイムを計算
        adjusted_marathon_time = ""
        if adjusted_target_vdot:
            adjusted_marathon_time = calculate_marathon_time_from_vdot(df_vdot, adjusted_target_vdot)
        
        # 過去遡り開始の説明（別途表示用）
        today = datetime.now()
        is_past_start = start_date < today
        
        # VDOT差チェックと警告
        if vdot_diff > 3.0 and adjusted_target_vdot:
            st.markdown(f"""
<div class="warning-box">
    <h4>⚠️ 目標タイムについての重要なお知らせ</h4>
    <p>現在のVDOT（<strong>{vdot_info['vdot']}</strong>）と入力された目標VDOT（<strong>{original_target_vdot}</strong>）の差が <strong>{vdot_diff}</strong> あります。</p>
    <p>VDOT差が3.0を超える場合、1つのトレーニングサイクル（約12〜16週間）で達成するのは難しい可能性があります。</p>
    <div style="border-top: 1px solid #FFB74D; margin-top: 1rem; padding-top: 1rem;">
    <h4>📊 今回のトレーニング計画について</h4>
    <p>そこで、今回のトレーニング計画では<strong>中間目標</strong>を設定します：</p>
    <ul>
        <li><strong>中間目標VDOT: {adjusted_target_vdot}</strong>（VDOT差 3.0）</li>
        <li><strong>中間目標マラソンタイム: {adjusted_marathon_time}</strong></li>
    </ul>
    <p>この中間目標を達成した後、次のトレーニングサイクルで最終目標（VDOT {original_target_vdot} / {user_data.get('target_time', '')}）を目指すことをお勧めします。</p>
    <p><strong>段階的なアプローチ</strong>により、怪我のリスクを減らし、着実にタイムを縮めていくことができます。</p>
    </div>
</div>
            """, unsafe_allow_html=True)
            
            # 過去遡り開始の説明を別途表示
            if is_past_start:
                st.markdown(f"""
<div class="warning-box">
    <h4>📅 トレーニング開始日について</h4>
    <p>本プログラムの仕様として、<strong>最低12週間</strong>のトレーニング期間を確保しています。</p>
    <p>レース日までの期間が12週間に満たないため、トレーニング開始日を <strong>{start_date.strftime('%Y/%m/%d')}（過去の日付）</strong> に設定しています。</p>
    <p>実際には本日から計画を参考にして、残りの期間でできる限りのトレーニングを行ってください。</p>
</div>
                """, unsafe_allow_html=True)
        elif vdot_diff > 3.0:
            st.markdown(f"""
<div class="warning-box">
    <h4>⚠️ 目標タイムについてのアドバイス</h4>
    <p>現在のVDOT（{vdot_info['vdot']}）と目標VDOT（{target_vdot['vdot']}）の差が <strong>{vdot_diff}</strong> あります。</p>
    <p>VDOT差が3.0を超える場合、1つのトレーニングサイクル（約12〜16週間）で達成するのは難しい可能性があります。</p>
    <p><strong>推奨:</strong> まずは中間目標（VDOT差1.5〜2.0程度）を設定し、段階的にタイムを縮めていくことをお勧めします。</p>
</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="success-box">
    <h4>✅ 目標設定は適切です</h4>
    <p>VDOT差 <strong>{vdot_diff}</strong> は、{training_weeks}週間のトレーニングで十分達成可能な範囲です。</p>
</div>
            """, unsafe_allow_html=True)
        
        # VDOT解説
        st.markdown("""
<div class="vdot-explanation">
    <h4>📖 VDOTとは</h4>
    <p>VDOTは、ジャック・ダニエルズ博士が考案した走力指標です。現在のタイムから算出され、適切なトレーニングペースを導き出すことができます。</p>
    <ul>
        <li><strong>E (Easy)</strong>: 会話ができる楽なペース。全体の70-80%をこのペースで。</li>
        <li><strong>M (Marathon)</strong>: フルマラソンの目標ペース。</li>
        <li><strong>T (Threshold)</strong>: 乳酸閾値ペース。20〜30分維持できる強度。</li>
        <li><strong>I (Interval)</strong>: インターバルペース。3〜5分維持できる強度。</li>
        <li><strong>R (Repetition)</strong>: 反復ペース。短い距離のスピード練習用。</li>
    </ul>
</div>
        """, unsafe_allow_html=True)
        
        with st.expander("📐 VDOT計算過程を確認"):
            st.code(vdot_info.get("calculation_log", "計算ログなし"))
            if pace_info and pace_info.get("calculation_log"):
                st.code(pace_info.get("calculation_log", ""))
        
        # フェーズ説明（調整済み目標VDOTを使用）
        phase_vdots = calculate_phase_vdots(vdot_info['vdot'], effective_target_vdot, 4)
        weeks_per_phase = training_weeks // 4
        
        st.markdown(f"""
<div class="phase-explanation">
    <h4>📈 4フェーズ構成（全{training_weeks}週間）</h4>
    <table style="width: 100%; border-collapse: collapse;">
        <tr style="background-color: #E3F2FD;">
            <th style="padding: 8px; text-align: left;">フェーズ</th>
            <th style="padding: 8px; text-align: left;">期間</th>
            <th style="padding: 8px; text-align: left;">目標VDOT</th>
            <th style="padding: 8px; text-align: left;">主な目的</th>
        </tr>
        <tr>
            <td style="padding: 8px;">フェーズ1（基礎構築期）</td>
            <td style="padding: 8px;">第1〜{weeks_per_phase}週</td>
            <td style="padding: 8px;">{phase_vdots[0]}</td>
            <td style="padding: 8px;">基礎体力の構築</td>
        </tr>
        <tr style="background-color: #F5F5F5;">
            <td style="padding: 8px;">フェーズ2（強化期）</td>
            <td style="padding: 8px;">第{weeks_per_phase+1}〜{weeks_per_phase*2}週</td>
            <td style="padding: 8px;">{phase_vdots[1]}</td>
            <td style="padding: 8px;">持久力・スピードの強化</td>
        </tr>
        <tr>
            <td style="padding: 8px;">フェーズ3（実践期）</td>
            <td style="padding: 8px;">第{weeks_per_phase*2+1}〜{weeks_per_phase*3}週</td>
            <td style="padding: 8px;">{phase_vdots[2]}</td>
            <td style="padding: 8px;">レースペースへの適応</td>
        </tr>
        <tr style="background-color: #F5F5F5;">
            <td style="padding: 8px;">フェーズ4（調整期）</td>
            <td style="padding: 8px;">第{weeks_per_phase*3+1}〜{training_weeks}週</td>
            <td style="padding: 8px;">{phase_vdots[3]}</td>
            <td style="padding: 8px;">テーパリング・最終調整</td>
        </tr>
    </table>
</div>
        """, unsafe_allow_html=True)
        
        # トレーニング計画生成
        if not st.session_state.training_plan:
            with st.spinner("🏃 トレーニング計画を作成中...（1〜2分程度かかります）"):
                try:
                    model = get_gemini_model()
                    if model:
                        # 調整済み目標VDOTを使用（あれば）
                        effective_target_vdot_for_prompt = {
                            "vdot": effective_target_vdot
                        }
                        prompt = create_training_prompt(
                            user_data, vdot_info, pace_info, effective_target_vdot_for_prompt, 
                            df_pace, training_weeks, start_date, df_vdot
                        )
                        response = model.generate_content(prompt)
                        # HTMLタグを除去
                        st.session_state.training_plan = sanitize_gemini_output(response.text)
                except Exception as e:
                    st.error(f"APIエラーが発生しました: {str(e)}")
                    st.session_state.training_plan = None
        
        # トレーニング計画表示
        if st.session_state.training_plan:
            st.markdown("---")
            st.markdown("## 📋 トレーニング計画")
            st.markdown(st.session_state.training_plan)
            
            # ダウンロードボタン
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                md_content = st.session_state.training_plan
                md_bytes = create_md_download(md_content)
                filename = f"training_plan_{user_data.get('name', 'user')}_{datetime.now().strftime('%Y%m%d')}.md"
                
                st.download_button(
                    label="📥 MDファイルをダウンロード",
                    data=md_bytes,
                    file_name=filename,
                    mime="text/markdown",
                    use_container_width=True
                )
            
            with col2:
                if st.button("🔄 計画を再生成", use_container_width=True):
                    st.session_state.training_plan = None
                    st.rerun()
            
            with col3:
                if st.button("📝 入力からやり直す", use_container_width=True):
                    st.session_state.form_submitted = False
                    st.session_state.training_plan = None
                    st.rerun()
        
        # フッター
        st.markdown("---")
        st.caption(f"{APP_NAME} v{APP_VERSION} | © 2025 VDOT Training System")


if __name__ == "__main__":
    main()
