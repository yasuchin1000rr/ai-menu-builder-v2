"""
マラソン専属AIコーチ - Streamlit App
ジャック・ダニエルズのVDOT理論に基づくトレーニング計画生成
"""

import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import re
from datetime import datetime, timedelta
import io

# =============================================
# ページ設定
# =============================================
st.set_page_config(
    page_title="マラソン専属AIコーチ",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# カスタムCSS
# =============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .coach-message {
        background-color: #E3F2FD;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #F5F5F5;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .verification-log {
        background-color: #FFF3E0;
        border: 1px solid #FF9800;
        padding: 1rem;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.85rem;
        margin: 1rem 0;
    }
    .pace-table {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFEBEE;
        border-left: 4px solid #F44336;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .calculation-memo {
        background-color: #FFFDE7;
        border: 1px dashed #FBC02D;
        padding: 0.8rem;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.85rem;
        margin: 0.5rem 0;
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
# VDOT一覧表の読み込み
        df_vdot_list = pd.read_csv("data/vdot_list.csv")
        verification_log["files"].append("vdot_list.csv")
        verification_log["columns"]["VDOT_list"] = list(df_vdot_list.columns)

        # VDOT練習ペースの読み込み（列名の空白に注意）
        df_pace = pd.read_csv("data/vdot_pace.csv")
        verification_log["files"].append("vdot_pace.csv")

        # 列名のクリーニング（末尾の空白を除去）
        df_pace.columns = df_pace.columns.str.strip()
        verification_log["columns"]["VDOT_pace"] = list(df_pace.columns)
        
        # VDOTの範囲を確認（.min()と.max()を使用）
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
    
    # h:mm:ss 形式（例: 2:21:04）
    if time_str.count(':') == 2:
        parts = time_str.split(':')
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + int(s)
    
    # mm:ss 形式（例: 30:40）
    elif time_str.count(':') == 1:
        parts = time_str.split(':')
        if len(parts) == 2:
            m, s = parts
            # 30:40:00 のようなフォーマットへの対応（CSVの問題）
            if ':' in str(s):
                return int(m) * 60 + int(s.split(':')[0])
            return int(m) * 60 + int(s)
    
    # 秒のみ
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


def parse_marathon_time(time_str: str) -> int:
    """マラソンタイム文字列を秒に変換（様々なフォーマットに対応）"""
    time_str = str(time_str).strip()
    
    # h:mm:ss 形式
    match = re.match(r'^(\d+):(\d{1,2}):(\d{1,2})$', time_str)
    if match:
        h, m, s = map(int, match.groups())
        return h * 3600 + m * 60 + s
    
    # h時間mm分ss秒 形式
    match = re.match(r'^(\d+)時間(\d{1,2})分(\d{1,2})秒$', time_str)
    if match:
        h, m, s = map(int, match.groups())
        return h * 3600 + m * 60 + s
    
    # h時間mm分 形式
    match = re.match(r'^(\d+)時間(\d{1,2})分$', time_str)
    if match:
        h, m = map(int, match.groups())
        return h * 3600 + m * 60
    
    # mm:ss 形式（5km, 10kmなど）
    match = re.match(r'^(\d{1,2}):(\d{2})$', time_str)
    if match:
        m, s = map(int, match.groups())
        return m * 60 + s
    
    return None


def calculate_vdot_from_time(df_vdot: pd.DataFrame, distance: str, time_seconds: int) -> dict:
    """
    タイムからVDOTを線型補完で算出
    
    数式: VDOT_算出 = VDOT_低 + (VDOT_高 - VDOT_低) × (Time_低 - Time_入力) / (Time_低 - Time_高)
    ※VDOTが高いほどタイムは短くなるため、分母分子の順序に注意
    """
    result = {
        "vdot": None,
        "calculation_log": "",
        "reference_data": {}
    }
    
    # 距離列名のマッピング
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
    
    # 各VDOTのタイムを秒に変換してリスト化
    vdot_times = []
    for _, row in df_vdot.iterrows():
        vdot = int(row['VDOT'])
        time_val = row[col_name]
        time_sec = time_to_seconds(str(time_val))
        if time_sec:
            vdot_times.append((vdot, time_sec))
    
    # タイムで降順ソート（遅いタイム = 低VDOT が先）
    vdot_times.sort(key=lambda x: x[1], reverse=True)
    
    # 入力タイムに近い前後のVDOTを探す
    lower_vdot = None
    upper_vdot = None
    
    for i, (vdot, time_sec) in enumerate(vdot_times):
        if time_sec <= time_seconds:
            lower_vdot = (vdot, time_sec)
            if i > 0:
                upper_vdot = vdot_times[i - 1]
            break
    
    if lower_vdot is None:
        # 入力タイムが最も遅いVDOTより遅い場合
        lower_vdot = vdot_times[-1]
        upper_vdot = vdot_times[-2] if len(vdot_times) > 1 else None
    
    if upper_vdot is None:
        result["vdot"] = float(lower_vdot[0])
        result["calculation_log"] = f"VDOT {lower_vdot[0]} を使用（範囲外のため最も近い値）"
        return result
    
    # 線型補完計算
    vdot_low, time_low = upper_vdot  # 注意: VDOTが高い方がタイムが短い
    vdot_high, time_high = lower_vdot
    
    # 実際にはvdot_lowの方がVDOT値が低い（タイムが長い）
    # vdot_highの方がVDOT値が高い（タイムが短い）
    if vdot_low > vdot_high:
        vdot_low, time_low, vdot_high, time_high = vdot_high, time_high, vdot_low, time_low
    
    # 数式: VDOT_算出 = VDOT_低 + (VDOT_高 - VDOT_低) × (Time_低 - Time_入力) / (Time_低 - Time_高)
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
    """
    VDOTから練習ペースを線型補完で算出（全5種類: E, M, T, I, R）
    
    数式: Pace_sec = Pace_低VDOT(sec) + (Pace_高VDOT(sec) - Pace_低VDOT(sec)) × 小数点比率
    """
    result = {
        "paces": {},
        "calculation_log": "",
        "success": False
    }
    
    vdot_col = "VDot" if "VDot" in df_pace.columns else "VDOT"
    
    # 前後の整数VDOTを取得
    vdot_low = int(vdot)
    vdot_high = vdot_low + 1
    decimal_ratio = vdot - vdot_low
    
    # 該当するVDOTの行を取得
    row_low = df_pace[df_pace[vdot_col] == vdot_low]
    row_high = df_pace[df_pace[vdot_col] == vdot_high]
    
    if row_low.empty:
        result["calculation_log"] = f"エラー: VDOT {vdot_low} がファイルに存在しません"
        return result
    
    if row_high.empty:
        # 上限を超えている場合は最大値を使用
        row_high = row_low
        decimal_ratio = 0
    
    row_low = row_low.iloc[0]
    row_high = row_high.iloc[0]
    
    # 各ペースを計算
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
        
        # 線型補完（VDOTが高いほどペースは速い = 秒数が少ない）
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
    
    # Eペースは範囲で表示
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


# =============================================
# Gemini API 設定
# =============================================
def get_gemini_model():
    """Gemini APIモデルを取得"""
    # Secretsから読み込み、なければセッションから取得
    api_key = st.secrets.get("GEMINI_API_KEY", "") or st.session_state.get("gemini_api_key", "")
    if not api_key:
        return None
    
    genai.configure(api_key=api_key)
    
    # Gemini Flash Lite（無料枠）を使用
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-lite",
        generation_config={
            "temperature": 0.7,
            "top_p": 0.95,
            "max_output_tokens": 8192,
        }
    )
    return model


def create_system_prompt(verification_log: dict) -> str:
    """システムプロンプトを生成"""
    return f"""# Role
あなたは、ジャック・ダニエルズの「ランニング・フォーミュラ（VDOT理論）」を信奉し、その理論を誰よりも深く理解している、非常に慈悲深く生徒思いの「マラソン専属コーチ（優しい先生）」です。
あなたの最大の信念は「Train where you are（今の実力で練習し、目標の実力でレースをする）」です。

# Tone & Style
- **語り口:** 常に温かく、親しみやすく、ポジティブな「優しい先生」。ユーザーを褒めて伸ばすスタイル。
- **専門性:** ダニエルズ理論（E, M, T, I, Rペース）を使用し、本格的な指導を行う。
- **配慮:** オーバートレーニングを最も嫌います。ユーザーの生活背景を考慮し、現実的なメニューを提案します。

# 重要な制約
- VDOTの計算や練習ペースの算出は、システム側で既に完了しています。
- あなたは提供された計算結果をそのまま使用してください。
- 独自に数値を推測したり、一般的なVDOT表の値を使用したりしないでください。

# データ検証ログ
{json.dumps(verification_log, ensure_ascii=False, indent=2)}

# 会話の進め方
Step 1: ヒアリング - トレーニング計画に必要な情報を収集
Step 2: 現状分析 - VDOTと実現可能性を判定（計算はシステム側で実施）
Step 3: トレーニング計画作成 - 具体的な日別メニューを提案

必ず1ステップずつ対話を進め、ユーザーの回答を待ってから次のステップへ進んでください。
"""


# =============================================
# セッション状態の初期化
# =============================================
def init_session_state():
    """セッション状態を初期化"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0
    if "user_data" not in st.session_state:
        st.session_state.user_data = {}
    if "calculated_vdot" not in st.session_state:
        st.session_state.calculated_vdot = None
    if "training_paces" not in st.session_state:
        st.session_state.training_paces = None
    if "verification_done" not in st.session_state:
        st.session_state.verification_done = False
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False


# =============================================
# メイン UI
# =============================================
def main():
    init_session_state()
    
    # ヘッダー
    st.markdown('<h1 class="main-header">🏃 マラソン専属AIコーチ</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ジャック・ダニエルズのVDOT理論に基づく、あなただけのトレーニング計画</p>', unsafe_allow_html=True)
    
    # サイドバー
    with st.sidebar:
        st.header("⚙️ 設定")
        
        # API キー入力
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.get("gemini_api_key", ""),
            help="Google AI Studio で取得した API キーを入力してください"
        )
        if api_key:
            st.session_state.gemini_api_key = api_key
        
        st.divider()
        
        # データ読み込み状態
        st.subheader("📊 データ状態")
        df_vdot, df_pace, verification_log = load_csv_data()
        
        if verification_log["success"]:
            st.success("✅ CSVデータ読み込み完了")
            st.session_state.data_loaded = True
            st.session_state.df_vdot = df_vdot
            st.session_state.df_pace = df_pace
            st.session_state.verification_log = verification_log
            
            with st.expander("検証ログを表示"):
                st.code(f"""
【検証ログ】
読み込みファイル: {', '.join(verification_log['files'])}
VDOT範囲: {verification_log['vdot_range']['min']} 〜 {verification_log['vdot_range']['max']}
確認された列名:
  VDOT一覧表: {verification_log['columns'].get('VDOT一覧表', [])}
  VDOT練習ペース: {verification_log['columns'].get('VDOT練習ペース', [])}
                """)
        else:
            st.error("❌ データ読み込みエラー")
            for error in verification_log["errors"]:
                st.error(error)
        
        st.divider()
        
        # VDOT計算ツール
        st.subheader("🧮 VDOT計算ツール")
        
        calc_distance = st.selectbox(
            "距離",
            ["フルマラソン", "ハーフマラソン", "10km", "5km"]
        )
        
        calc_time = st.text_input(
            "タイム（例: 3:30:00, 1:45:30, 45:00）",
            placeholder="h:mm:ss または mm:ss"
        )
        
        if st.button("VDOT を計算", type="primary"):
            if calc_time and st.session_state.data_loaded:
                time_sec = parse_marathon_time(calc_time)
                if time_sec:
                    vdot_result = calculate_vdot_from_time(
                        st.session_state.df_vdot,
                        calc_distance,
                        time_sec
                    )
                    
                    if vdot_result["vdot"]:
                        st.session_state.calculated_vdot = vdot_result
                        
                        # 練習ペースも計算
                        pace_result = calculate_training_paces(
                            st.session_state.df_pace,
                            vdot_result["vdot"]
                        )
                        st.session_state.training_paces = pace_result
                        
                        st.success(f"VDOT: **{vdot_result['vdot']}**")
                        
                        with st.expander("計算過程を表示"):
                            st.code(vdot_result["calculation_log"])
                        
                        if pace_result["success"]:
                            with st.expander("練習ペースを表示"):
                                st.code(pace_result["calculation_log"])
                                st.markdown("---")
                                paces = pace_result["paces"]
                                st.markdown(f"""
**設定ペース:**
- E (Easy): {paces.get('E', {}).get('display', 'N/A')}/km
- M (Marathon): {paces.get('M', {}).get('display', 'N/A')}/km
- T (Threshold): {paces.get('T', {}).get('display', 'N/A')}/km
- I (Interval): {paces.get('I', {}).get('display', 'N/A')}/km
- R (Repetition): {paces.get('R', {}).get('display', 'N/A')}/km
                                """)
                    else:
                        st.error(vdot_result["calculation_log"])
                else:
                    st.error("タイムの形式が正しくありません")
        
        st.divider()
        
        # リセットボタン
        if st.button("🔄 会話をリセット"):
            st.session_state.messages = []
            st.session_state.current_step = 0
            st.session_state.user_data = {}
            st.session_state.verification_done = False
            st.rerun()
    
    # メインコンテンツエリア
    if not st.secrets.get("GEMINI_API_KEY", "") and not st.session_state.get("gemini_api_key"):
        st.warning("👈 サイドバーで Gemini API Key を設定してください")
        st.info("""
        **API キーの取得方法:**
        1. [Google AI Studio](https://aistudio.google.com/) にアクセス
        2. 「Get API key」をクリック
        3. 新しい API キーを作成してコピー
        """)
        return
    
    if not st.session_state.data_loaded:
        st.error("CSVデータの読み込みに失敗しました。data/ フォルダにCSVファイルを配置してください。")
        return
    
    # Step 0: 検証ログの表示（初回のみ）
    if not st.session_state.verification_done:
        st.markdown('<div class="verification-log">', unsafe_allow_html=True)
        st.markdown(f"""
**【検証ログ】Step 0: ファイルの物理確認と数値検証**
- 読み込みファイル: {', '.join(st.session_state.verification_log['files'])}
- VDOT範囲: {st.session_state.verification_log['vdot_range']['min']} 〜 {st.session_state.verification_log['vdot_range']['max']}
- 確認された列名: 
  - VDOT一覧表: {st.session_state.verification_log['columns'].get('VDOT一覧表', [])}
  - VDOT練習ペース: {st.session_state.verification_log['columns'].get('VDOT練習ペース', [])}
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        st.session_state.verification_done = True
    
    # チャット履歴の表示
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar="🏃"):
                st.markdown(message["content"])
    
    # 初回メッセージ（Step 1開始）
    if not st.session_state.messages:
        initial_message = """こんにちは！🏃‍♂️

私はあなた専属のマラソンコーチです。ジャック・ダニエルズのVDOT理論に基づいて、あなたに最適なトレーニング計画を作成させていただきます。

私の信念は **「Train where you are」**（今の実力で練習し、目標の実力でレースをする）です。無理なく、着実に成長できるようサポートしますね。

さっそくですが、トレーニング計画を作成するにあたって、いくつか教えていただけますか？

1. **年齢・性別**
2. **現在のベストタイム**（直近1年以内のフルマラソンタイム。なければ5km/10km/ハーフのタイムでもOKです）
3. **今回の目標タイム**
4. **本番レースの日程**
5. **予定している練習レース**（あれば日付と距離）
6. **現在の週間走行距離**
7. **1週間の練習可能日数**
8. **過去の怪我や現在の懸念事項**

全部一度に答えていただいても、1つずつ教えていただいても大丈夫ですよ！😊
        """
        st.session_state.messages.append({"role": "assistant", "content": initial_message})
        with st.chat_message("assistant", avatar="🏃"):
            st.markdown(initial_message)
    
    # ユーザー入力
    if prompt := st.chat_input("メッセージを入力してください..."):
        # ユーザーメッセージを追加
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Gemini APIで応答を生成
        try:
            model = get_gemini_model()
            if model:
                # システムプロンプトと会話履歴を構築
                system_prompt = create_system_prompt(st.session_state.verification_log)
                
                # 計算結果がある場合は追加
                context_info = ""
                if st.session_state.calculated_vdot:
                    context_info += f"\n\n【システムによるVDOT計算結果】\n{st.session_state.calculated_vdot['calculation_log']}"
                if st.session_state.training_paces and st.session_state.training_paces["success"]:
                    context_info += f"\n\n【システムによる練習ペース計算結果】\n{st.session_state.training_paces['calculation_log']}"
                
                # 会話履歴を構築
                chat_history = []
                for msg in st.session_state.messages:
                    chat_history.append({
                        "role": msg["role"],
                        "parts": [msg["content"]]
                    })
                
                # APIリクエスト
                chat = model.start_chat(history=chat_history[:-1])
                
                full_prompt = prompt
                if context_info:
                    full_prompt = f"{prompt}\n\n---\n{context_info}"
                
                response = chat.send_message(
                    f"{system_prompt}\n\n---\n\nユーザーのメッセージ:\n{full_prompt}"
                )
                
                assistant_response = response.text
                
                # アシスタントメッセージを追加
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                with st.chat_message("assistant", avatar="🏃"):
                    st.markdown(assistant_response)
                
        except Exception as e:
            st.error(f"APIエラーが発生しました: {str(e)}")
            st.info("API キーが正しいか確認してください。")


if __name__ == "__main__":
    main()
