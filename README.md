# 🏃 マラソン専属AIコーチ

ジャック・ダニエルズの **VDOT理論** に基づく、パーソナライズされたマラソントレーニング計画を生成するAIコーチアプリケーションです。

## ✨ 特徴

- **VDOT理論の完全実装**: ダニエルズのランニング・フォーミュラを忠実に再現
- **線型補完による正確な計算**: 整数VDOT間のタイムも正確に算出
- **5種類の練習ペース自動算出**: E(Easy), M(Marathon), T(Threshold), I(Interval), R(Repetition)
- **計算過程の透明化**: 全ての計算過程をログとして表示
- **優しいコーチ**: 温かく励ましながら現実的なトレーニング計画を提案

## 🚀 デプロイ方法

### Streamlit Cloud へのデプロイ

1. **GitHubリポジトリを作成**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. **Streamlit Cloud で設定**
   - [Streamlit Cloud](https://streamlit.io/cloud) にアクセス
   - GitHubリポジトリを連携
   - `app.py` をメインファイルとして指定

3. **Secrets の設定**（任意）
   - Streamlit Cloud の Settings > Secrets で API キーを設定可能
   ```toml
   GEMINI_API_KEY = "your-api-key-here"
   ```

## 📁 ファイル構成

```
├── app.py                    # メインアプリケーション
├── requirements.txt          # Python依存パッケージ
├── README.md                 # このファイル
└── data/
    ├── VDOT一覧表.csv         # 距離別VDOTタイム対応表
    └── VDOT練習ペース.csv     # VDOT別練習ペース表
```

## 🔧 ローカル開発

### 必要要件

- Python 3.9+
- Gemini API Key（[Google AI Studio](https://aistudio.google.com/) で取得）

### セットアップ

```bash
# リポジトリをクローン
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 仮想環境を作成（推奨）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存パッケージをインストール
pip install -r requirements.txt

# アプリを起動
streamlit run app.py
```

## 📊 VDOT計算の仕組み

### VDOTの線型補完

入力タイムが整数VDOT間にある場合、以下の数式で算出:

```
VDOT_算出 = VDOT_低 + (VDOT_高 - VDOT_低) × (Time_低 - Time_入力) / (Time_低 - Time_高)
```

※VDOTが高いほどタイムは短くなるため、分母分子の順序に注意

### 練習ペースの線型補完

小数点を含むVDOTの練習ペースを算出:

```
Pace_sec = Pace_低VDOT(sec) + (Pace_高VDOT(sec) - Pace_低VDOT(sec)) × 小数点比率
```

## 🤖 使用API

- **運用**: Google Gemini Flash Lite（無料枠）
- **モデル**: `gemini-2.0-flash-lite`

## 📋 コーチのプロセス

### Step 0: ファイル検証（自動）
- CSVファイルの読み込みと整合性チェック
- VDOT範囲と列名の確認
- 検証ログの表示

### Step 1: ヒアリング
- 年齢・性別
- 現在のベストタイム
- 目標タイム
- レース日程
- 週間走行距離
- 練習可能日数
- 怪我の履歴

### Step 2: 現状分析
- VDOTの算出（計算過程を表示）
- 実現可能性の判定
- 必要に応じた下方修正の提案
- フェーズ別ステップアップ計画

### Step 3: トレーニング計画作成
- 日別メニューの作成
- 全5種類のペース設定
- 週間テーマと想定距離
- コーチからのアドバイス

## ⚠️ 注意事項

- VDOTの計算は添付CSVファイルのデータに基づきます
- 一般的なVDOT表と異なる数値の場合でもファイル側を正とします
- 怪我のリスクを最小限にするため、段階的なVDOT引き上げを推奨

## 📝 ライセンス

MIT License

## 🙏 クレジット

- VDOT理論: ジャック・ダニエルズ「ランニング・フォーミュラ」
- AI: Google Gemini API
- フレームワーク: Streamlit
