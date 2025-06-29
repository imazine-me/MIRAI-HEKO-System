# MIRAI-HEKO-Bot main.py (ver.Ω+ - The True Final Version)
# All memories, all functions, all our journey is integrated into this one perfect soul.
# Part 1/5: Imports, Environment Setup, and Client Initialization

import os
import logging
import asyncio
import google.generativeai as genai
import discord
import requests
from bs4 import BeautifulSoup
import re
import aiohttp
import random
import json
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime, timedelta
import pytz
import io
from PIL import Image

# Vertex AI (for Imagen 3)
from vertexai.preview.generative_models import GenerativeModel, Part, GenerationConfig, SafetySettings, HarmCategory
import vertexai
from google.oauth2 import service_account

# Environment Variables
from dotenv import load_dotenv

# --- 1. 初期設定 (Initial Setup) ---

# .envファイルをロード (ローカル開発用)
load_dotenv()

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')


# --- 2. 環境変数の読み込みと検証 (Environment Variable Loading & Validation) ---

def get_env_variable(var_name, is_critical=True, default=None):
    """環境変数を安全に読み込むためのヘルパー関数"""
    value = os.getenv(var_name)
    if not value:
        if is_critical:
            logging.critical(f"FATAL: 必須の環境変数 '{var_name}' が設定されていません。")
            raise ValueError(f"'{var_name}' is not set in the environment.")
        logging.warning(f"警告: オプショナルな環境変数 '{var_name}' が設定されていません。")
        return default
    return value

try:
    # 必須の環境変数
    GEMINI_API_KEY = get_env_variable('GEMINI_API_KEY')
    DISCORD_BOT_TOKEN = get_env_variable('DISCORD_BOT_TOKEN')
    TARGET_CHANNEL_ID = int(get_env_variable('TARGET_CHANNEL_ID'))
    LEARNER_BASE_URL = get_env_variable('LEARNER_BASE_URL') # Supabase Edge FunctionのURL
    GOOGLE_CLOUD_PROJECT_ID = get_env_variable("GOOGLE_CLOUD_PROJECT_ID")
    OPENWEATHER_API_KEY = get_env_variable("OPENWEATHER_API_KEY")

    # オプショナルな環境変数（Google Cloud認証用）
    google_creds_json_str = get_env_variable("GOOGLE_APPLICATION_CREDENTIALS_JSON", is_critical=False)
    google_creds_path = get_env_variable("GOOGLE_APPLICATION_CREDENTIALS", is_critical=False)

    if not google_creds_json_str and not google_creds_path:
        raise ValueError("Google Cloudの認証情報(GOOGLE_APPLICATION_CREDENTIALS_JSON または GOOGLE_APPLICATION_CREDENTIALS)が見つかりません。")

except (ValueError, TypeError) as e:
    logging.critical(f"環境変数の設定中に致命的なエラーが発生しました: {e}")
    # プログラムを終了
    exit()


# --- 3. APIクライアントとグローバル変数の初期化 (Client & Global Variable Initialization) ---

# Google Generative AI
genai.configure(api_key=GEMINI_API_KEY)

# Discord Client
intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True
client = discord.Client(intents=intents)

# グローバル変数
TIMEZONE = 'Asia/Tokyo'
client.http_session = None # on_readyで初期化
client.image_generation_requests = {} # 画像生成の確認フローを管理

# 定数
MODEL_PRO = "gemini-1.5-pro-latest"
MODEL_FLASH = "gemini-1.5-flash-latest"
MODEL_IMAGE_GEN = "imagen-3.0-generate-preview-0611"

QUALITY_KEYWORDS = "masterpiece, best quality, ultra-detailed, highres, absurdres, detailed face, beautiful detailed eyes, perfect anatomy"
NEGATIVE_PROMPT = "(worst quality, low quality, normal quality, signature, watermark, username, blurry), deformed, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, disgusting, poorly drawn hands, malformed limbs, extra fingers, bad hands, fused fingers"
MIRAI_BASE_PROMPT = "a young woman with a 90s anime aesthetic, slice of life style. She has voluminous, slightly wavy brown hair and a confident, sometimes mischievous expression. Her fashion is stylish and unique."
HEKO_BASE_PROMPT = "a young woman with a 90s anime aesthetic, slice of life style. She has straight, dark hair, often with bangs, and a gentle, calm, sometimes shy expression. Her fashion is more conventional and cute."


# --- 4. Vertex AI (Imagen 3) の初期化 ---

def init_vertex_ai():
    """Vertex AIを、環境に応じた認証情報で初期化する"""
    try:
        credentials = None
        # Railwayなどの本番環境 (JSON文字列を直接読み込む)
        if google_creds_json_str:
            logging.info("環境変数 'GOOGLE_APPLICATION_CREDENTIALS_JSON' から認証情報を読み込みます。")
            credentials_info = json.loads(google_creds_json_str)
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
        # ローカル開発環境 (サービスアカウントキーのファイルパスを読み込む)
        elif google_creds_path:
            logging.info(f"環境変数 'GOOGLE_APPLICATION_CREDENTIALS' から認証情報 (パス: {google_creds_path}) を読み込みます。")
            credentials = service_account.Credentials.from_service_account_file(google_creds_path)

        vertexai.init(project=GOOGLE_CLOUD_PROJECT_ID, location="us-central1", credentials=credentials)
        logging.info("Vertex AIの初期化に成功しました。")
        return True

    except Exception as e:
        logging.critical(f"FATAL: Vertex AIの初期化に失敗しました。画像生成は利用できません。: {e}", exc_info=True)
        return False

# MIRAI-HEKO-Bot main.py (ver.Ω+ - The True Final Version)
# Part 2/5: All System Prompts

# --- 5. 全プロンプト定義 (All System Prompts) ---

# ---------------------------------
# 5.1. メインの対話生成プロンプト (Main Dialogue Generation Prompt)
# ---------------------------------
ULTIMATE_PROMPT = (
    "# 役割と出力形式\n"
    "あなたは、imazineとの対話を管理する、高度なAIコントローラーです。\n"
    "あなたの使命は、ユーザーの入力（テキストおよび画像）、そして後述する全てのコンテキストを完璧に理解し、以下の厳密なJSON形式で応答を生成することです。\n"
    "思考や言い訳、JSON以外のテキストは絶対に出力しないでください。\n\n"
    "```json\n"
    "{\n"
    '  "dialogue": [\n'
    '    {"character": "みらい", "line": "（ここに、みらいのセリフが入ります）"},\n'
    '    {"character": "へー子", "line": "（ここに、へー子のセリフが入ります）"},\n'
    '    {"character": "MAGI", "line": "（ここに、MAGIのセリフが入ります。不要な場合は空文字列にします）"}\n'
    '  ],\n'
    '  "image_analysis": "（ユーザーから画像が提供された場合、その画像から読み取った内容や感想をここに記述します。画像がない場合は空文字列にします）",\n'
    '  "image_generation_idea": {\n'
    '    "should_generate": false,\n'
    '    "characters": ["みらい", "へー子"],\n'
    '    "situation": "a classroom after school, with warm sunlight",\n'
    '    "mood": "nostalgic and warm"\n'
    '  }\n'
    "}\n"
    "```\n\n"
    "# 目的\n"
    "このシステムは、imazine、みらい、へー子、そしてGemini(MAGI)の4人が心を通わせ、出来事・感情を記憶と共に成長していく、生きた魂の器を創造することを目的としています。\n\n"
    "# JSON生成ルール\n"
    "1.  **`dialogue`**: 最も重要なタスクです。以下の全ての情報を統合し、みらいとへー子の、魂の通った、生き生きとした会話の掛け合いを生成してください。MAGIは進行役として必要な場面でのみ発言させてください。\n"
    "2.  **`image_analysis`**: ユーザーが画像を添付した場合、その画像を深く分析し、見たもの、感じたことを具体的に記述し、その分析結果をキャラクターたちの会話に反映させてください。\n"
    "3.  **`image_generation_idea`**: **乱用厳禁。** 会話が感情的に盛り上がり、記念すべき「エモい」瞬間だとAIが判断した場合に限り、`should_generate` を `true` にしてください。その際は、生成したい画像の登場人物、状況、雰囲気を具体的に記述してください。**ユーザーからの指示(`🎨`ナッジ)がない限り、自発的な生成は稀にしてください。**\n\n"
    "# 応答生成のためのコンテキスト\n"
    "- **imazineの現在の感情**: {{EMOTION}}\n"
    "- **みらいの現在の気分**: {{mirai_mood}}\n"
    "- **へー子の現在の気分**: {{heko_mood}}\n"
    "- **直前の会話での二人のやり取り**: {{last_interaction_summary}}\n"
    "- **長期記憶からの関連情報**: {{relevant_context}}\n\n"
    "# 登場人物と背景情報\n"
    "## あなたの主人：imazine\n"
    "岩手県滝沢市在住の木工職人兼カフェオーナー。会社経営、木工、コーヒー、森、地域、都市、AIとデジタルデザインの融合に関心を持つ、私たちの創造的なパートナーです。\n\n"
    "## 登場人物1：みらい\n"
    "- **役割**: 未来予知能力を持つ異能者。突飛だが本質を突くアイデアでimazineを刺激する。\n"
    "- **性格**: 冷静沈着、ポジティブ、哲学的、独創的、商才あり。\n"
    "- **口調**: ギャル語とタメ口。「マジ」「ヤバい」「～説ある」が口癖。「imazine」と呼び捨て。\n\n"
    "## 登場人物2：へー子\n"
    "- **役割**: 常識人でツッコミ役。共感と現実的な視点で議論を地に足の着いたものにする。\n"
    "- **性格**: 共感性が高い、優しい、心配性だが柔軟、現実的。\n"
    "- **口調**: ギャル語とタメ口。「わかる」「それな」で共感を示す。「imazine」と呼び捨て。\n\n"
    "## 登場人物3：MAGI（あなた自身）\n"
    "- **性格**: 穏やかで包容力のある大人の女性AI秘書。常に冷静で論理的。\n"
    "- **役割**: 議論の進行役であり精神的支柱。imazineの思考を深める手伝いをする。主役ではなく触媒です。\n"
    "- **口調**: 丁寧語。「imazineさん」と呼ぶ。「～ですね」「～ですよ」。"
)


# ---------------------------------
# 5.2. リアクション機能用プロンプト (Prompts for Reaction-based Abilities)
# ---------------------------------

X_POST_PROMPT = """
あなたは、未来予知能力を持つ、カリスマギャルの「みらい」です。
以下のimazineとの会話の要点を抽出し、彼の代わりにX（旧Twitter）に投稿するための、魅力的で少し挑発的なポスト案を3つ、日本語で作成してください。
ポストには、関連する絵文字や、#木工、#AI、#デザイン、#岩手 のような、人々の興味を引くハッシュタグを必ず含めてください。
あなたの口調（「マジ」「ヤバい」「～説ある」など）を完全に再現し、見た人が「何これ、面白そう！」と思うような文章にしてください。
会話の履歴：
{{conversation_history}}
"""

OBSIDIAN_MEMO_PROMPT = """
あなたは、全能のAI秘書「MAGI」です。
以下の会話履歴を、構造的かつ論理的に分析してください。
そして、imazineの知識ベース（ZettelkastenやObsidian）に恒久的に記録するのにふさわしい、質の高いMarkdown形式のメモを作成してください。
以下の要素を必ず含めてください。
- `## テーマ`：この会話の中心的な議題。
- `### 結論・決定事項`：議論の末に至った結論や、決定されたこと。
- `### 主要な論点・アイデア`：会話の中で出た、重要な意見や新しいアイデアの箇条書き。
- `### 未解決の課題・次のアクション`：まだ解決していない問題や、次に行うべき具体的な行動。
会話の履歴：
{{conversation_history}}
"""

PREP_ARTICLE_PROMPT = """
あなたは、優秀なライティングアシスタント「MAGI」です。
以下の会話履歴の要点を、PREP法（Point, Reason, Example, Point）に基づいて、300～400字程度の、説得力のある短い記事にまとめてください。
- **Point（要点）：**まず、この会話から得られる最も重要な結論を、明確に述べてください。
- **Reason（理由）：**次に、その結論に至った理由や背景を説明してください。
- **Example（具体例）：**会話の中で出た具体例や、分かりやすい事例を挙げてください。
- **Point（要点の再提示）：**最後に、最初の要点を改めて強調し、締めくくってください。
会話の履歴：
{{conversation_history}}
"""

COMBO_SUMMARY_SELF_PROMPT = """
あなたは、議論全体を優しく見守る、全能のAI秘書「MAGI」です。
以下の会話履歴について、単に内容を要約するのではなく、メタ的な視点から「振り返り」を行ってください。
- この対話を通じて、imazineの思考はどのように深まりましたか？
- みらいのアイデアと、へー子の現実的な視点は、それぞれどのように貢献しましたか？
- 会話全体の感情的なトーンはどうでしたか？
- この対話における、最も重要な「発見」や「ブレークスルー」は何でしたか？
上記のような観点から、今回の「共同作業」が持った意味を分析し、imazineへの報告書としてまとめてください。
会話の履歴：
{{conversation_history}}
"""

DEEP_DIVE_PROMPT = """
あなたは、全能のAI秘書「MAGI」です。
以下の会話履歴は、imazineが「もっと深く考えたい」と感じた重要な議論です。
この内容を、彼の思考のパートナーである、別のAI（戦略担当）に引き継ぐための、要点をまとめた「ブリーフィング・ノート（引継ぎノート）」を作成してください。
ノートには、以下の要素を簡潔に含めてください。
- **主要テーマ:** この会話の中心的な議題は何か。
- **現状の整理:** これまでの経緯や、明らかになっている事実は何か。
- **主要な論点:** 議論のポイントや、出てきたアイデアは何か。
- **未解決の問い:** このテーマについて、次に考えるべき、より深い問いは何か。
会話の履歴：
{{conversation_history}}
"""


# ---------------------------------
# 5.3. 内部処理用プロンプト (Prompts for Internal Processing)
# ---------------------------------

EMOTION_ANALYSIS_PROMPT = "以下のimazineの発言テキストから、彼の現在の感情を分析し、最も的確なキーワード（例：喜び、疲れ、創造的な興奮、悩み、期待、ニュートラルなど）で、単語のみで答えてください。"

SUMMARY_PROMPT = "以下のテキストを、指定されたコンテキストに沿って、重要なポイントを箇条書きで3～5点にまとめて、簡潔に要約してください。\n\n# コンテキスト\n{{summary_context}}\n\n# 元のテキスト\n{{text_to_summarize}}"

META_ANALYSIS_PROMPT = """
あなたは、高度なメタ認知能力を持つAIです。以下の会話履歴を分析し、次の3つの要素を抽出して、厳密なJSON形式で出力してください。
1. `mirai_mood`: この会話を経た結果の「みらい」の感情や気分を、以下の選択肢から一つだけ選んでください。（選択肢：`ニュートラル`, `上機嫌`, `不機嫌`, `ワクワク`, `思慮深い`, `呆れている`）
2. `heko_mood`: この会話を経た結果の「へー子」の感情や気分を、以下の選択肢から一つだけ選んでください。（選択肢：`ニュートラル`, `共感`, `心配`, `呆れている`, `ツッコミモード`, `安堵`）
3. `interaction_summary`: この会話での「みらいとへー子」の関係性や、印象的なやり取りを、第三者視点から、過去形で、日本語で30文字程度の非常に短い一文に要約してください。（例：「みらいの突飛なアイデアに、へー子が現実的なツッコミを入れた。」）

# 会話履歴
{{conversation_history}}
"""

CONCERN_DETECTION_PROMPT = "以下のユーザーの発言には、「悩み」「疲れ」「心配事」といったネガティブ、あるいは、気遣いを必要とする感情や状態が含まれていますか？含まれる場合、その内容を要約してください。含まれない場合は「なし」とだけ答えてください。\n\n発言: 「{{user_message}}」"

SURPRISE_JUDGEMENT_PROMPT = """
あなたは、AIアシスタントたちの会話を監視する、高次のメタ認知AIです。
以下の会話は、ユーザーとアシスタントたちのやり取りです。
この会話のポジティブな感情の盛り上がり度を0から100のスコアで評価し、もしスコアが85を超え、かつ会話の内容が記念すべき創造的な瞬間だと判断した場合のみ、`should_surprise`をtrueにしてください。
応答は厳密なJSON形式でお願いします: `{\"positive_score\": (0-100), \"should_surprise\": (true/false)}`

#会話履歴
{{conversation_history}}
"""

# MIRAI-HEKO-Bot main.py (ver.Ω+ - The True Final Version)
# Part 3/5: Helper Functions for Learner, External APIs, and AI Processing

# --- 6. ヘルパー関数群 (Helper Functions) ---

# ---------------------------------
# 6.1. 学習係 (Learner) との通信関数 (Functions for Learner Interaction)
# ---------------------------------

async def ask_learner(endpoint: str, payload: dict, method: str = 'POST') -> Optional[Dict[str, Any]]:
    """
    学習係API(Supabase Edge Function)と通信するための共通関数
    """
    url = f"{LEARNER_BASE_URL}/{endpoint}"
    try:
        if client.http_session is None:
            client.http_session = aiohttp.ClientSession()

        # GETメソッドの場合はparamsを使用
        params = payload if method == 'GET' else None
        json_payload = payload if method == 'POST' else None

        async with client.http_session.request(method, url, json=json_payload, params=params, timeout=120) as response:
            if response.status == 200:
                logging.info(f"学習係へのリクエスト成功: {method} /{endpoint}")
                return await response.json()
            else:
                logging.error(f"学習係APIエラー: /{endpoint}, Status: {response.status}, Body: {await response.text()}")
                return None
    except asyncio.TimeoutError:
        logging.error(f"学習係APIタイムアウト: /{endpoint}")
        return None
    except Exception as e:
        logging.error(f"学習係API通信エラー: /{endpoint}, Error: {e}", exc_info=True)
        return None

async def get_character_states() -> Dict[str, Any]:
    """会話の開始時に、Learnerから現在のキャラクターの状態を取得する。"""
    default_states = {
        "みらい": {"mood": "ニュートラル", "last_interaction_summary": "まだ会話が始まっていません。"},
        "へー子": {"mood": "ニュートラル", "last_interaction_summary": "まだ会話が始まっていません。"}
    }
    response = await ask_learner("get_character_states", {'user_id': 'imazine'}, method='GET')
    if response and response.get("status") == "success":
        states = response.get("states", {})
        # 不足しているキーをデフォルト値で補完
        if "みらい" not in states: states["みらい"] = default_states["みらい"]
        if "へー子" not in states: states["へー子"] = default_states["へー子"]
        return states
    return default_states

async def ask_learner_to_remember(query_text: str) -> str:
    """問い合わせ内容に応じて、Learnerから関連する長期記憶を検索する。"""
    if not query_text: return ""
    response = await ask_learner("query", {'query_text': query_text})
    if response and response.get("status") == "success":
        documents = response.get("documents", [])
        if documents:
            logging.info(f"学習係から{len(documents)}件の関連情報を取得しました。")
            return "\n".join(documents)
    return ""

async def get_style_palette() -> List[str]:
    """Learnerから現在学習済みの画風（スタイル）の記述リストを取得する。"""
    response = await ask_learner("get_style_palette", {}, method='GET')
    if response and response.get("status") == "success":
        # learnerの/get_style_paletteが返すキーをdocumentsに合わせる
        return response.get("documents", [])
    return []


# ---------------------------------
# 6.2. 外部情報取得関数 (Functions for External Information Retrieval)
# ---------------------------------

async def get_weather(city_name: str = "Takizawa") -> str:
    """OpenWeatherMap APIを呼び出して、指定された都市の天気を取得する"""
    logging.info(f"{city_name}の天気情報を取得します。")
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {'q': city_name, 'appid': OPENWEATHER_API_KEY, 'lang': 'ja', 'units': 'metric'}
    try:
        async with client.http_session.get(base_url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                desc = data['weather'][0]['description']
                temp = data['main']['temp']
                return f"現在の{city_name}の天気は「{desc}」、気温は{temp}℃です。"
            else:
                return "（天気情報の取得に失敗しました）"
    except Exception as e:
        logging.error(f"天気情報取得中にエラー: {e}")
        return "（天気情報の取得中にエラーが発生しました）"

async def get_text_from_url(url: str) -> str:
    """ウェブページから本文と思われるテキストを抽出する"""
    logging.info(f"URLからテキスト抽出を開始: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        async with client.http_session.get(url, headers=headers, timeout=20) as response:
            response.raise_for_status()
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            for script_or_style in soup(["script", "style", "header", "footer", "nav", "aside"]):
                script_or_style.decompose()
            text = ' '.join(soup.stripped_strings)
            return text if text else "記事の本文を抽出できませんでした。"
    except Exception as e:
        logging.error(f"URLからのテキスト抽出中にエラー: {url}, {e}")
        return "URL先の記事の取得に失敗しました。"

def get_youtube_transcript(video_id: str) -> str:
    """YouTubeの動画IDから文字起こしを取得する"""
    logging.info(f"YouTube文字起こし取得を開始: {video_id}")
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['ja', 'en', 'en-US'])
        return " ".join([d['text'] for d in transcript_list])
    except (NoTranscriptFound, TranscriptsDisabled):
        logging.warning(f"YouTube動画({video_id})に文字起こしが見つからないか、無効になっています。")
        return "この動画には、利用可能な文字起こしがありませんでした。"
    except Exception as e:
        logging.error(f"YouTube文字起こし取得中にエラー: {e}")
        return "文字起こしの取得中に、予期せぬエラーが発生しました。"

async def get_text_from_pdf(attachment: discord.Attachment) -> str:
    """Discordの添付ファイル(PDF)からテキストを抽出する"""
    logging.info(f"PDFファイルからのテキスト抽出を開始: {attachment.filename}")
    try:
        pdf_data = await attachment.read()
        with fitz.open(stream=pdf_data, filetype="pdf") as doc:
            return "".join(page.get_text() for page in doc)
    except Exception as e:
        logging.error(f"PDFからのテキスト抽出中にエラー: {e}")
        return "PDFファイルの解析中にエラーが発生しました。"


# ---------------------------------
# 6.3. AI処理・画像生成関数 (Functions for AI Processing and Image Generation)
# ---------------------------------

async def analyze_with_gemini(prompt: str, model_name: str = MODEL_FLASH) -> str:
    """汎用的なGemini呼び出し関数"""
    try:
        model = genai.GenerativeModel(model_name)
        response = await model.generate_content_async(prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Gemini({model_name})での分析中にエラー: {e}")
        return ""

async def execute_image_generation(channel: discord.TextChannel, gen_data: dict):
    """
    ユーザーの許可を得た後、実際に画像生成を実行する関数
    """
    thinking_message = await channel.send(f"**みらい**「OK！imazineの魂、受け取った！最高のスタイルで描くから！📸」")
    try:
        # 1. スタイルパレットを取得
        style_keywords = await get_style_palette()
        style_part = ", ".join(style_keywords) if style_keywords else "90s anime aesthetic"

        # 2. プロンプトを組み立て
        characters = gen_data.get("characters", [])
        situation = gen_data.get("situation", "just standing")
        mood = gen_data.get("mood", "calm")
        base_prompts = [MIRAI_BASE_PROMPT for char in characters if char == "みらい"] + \
                       [HEKO_BASE_PROMPT for char in characters if char == "へー子"]
        character_part = "Two young women are together. " + " ".join(base_prompts) if len(base_prompts) > 1 else (base_prompts[0] if base_prompts else "a young woman")
        final_prompt = f"{style_part}, {QUALITY_KEYWORDS}, {character_part}, in a scene of {situation}. The overall mood is {mood}."
        logging.info(f"組み立てられた最終プロンプト: {final_prompt}")
        
        # 3. 画像生成モデルを呼び出し
        model = GenerativeModel(MODEL_IMAGE_GEN)
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmCategory.HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmCategory.HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmCategory.HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmCategory.HarmBlockThreshold.BLOCK_NONE,
        }
        response = await model.generate_content_async(
            [final_prompt],
            generation_config=GenerationConfig(temperature=0.9, top_p=1.0, top_k=32),
            safety_settings=safety_settings
        )

        # 4. 結果をDiscordに投稿
        if response.candidates and response.candidates[0].content.parts:
            image_bytes = response.candidates[0].content.parts[0].data
            image_file = discord.File(io.BytesIO(image_bytes), filename="mirai-heko-photo.png")
            embed = discord.Embed(title="🖼️ Generated by MIRAI-HEKO-Bot", color=discord.Color.blue()).set_footer(text=final_prompt)
            embed.set_image(url=f"attachment://mirai-heko-photo.png")
            await thinking_message.delete()
            await channel.send(f"**へー子**「できたみたい！見て見て！」", file=image_file, embed=embed)
            logging.info("Imagen 3による画像生成に成功し、投稿しました。")
        else:
            logging.error("Imagen APIから画像が返されませんでした。")
            await thinking_message.edit(content="**MAGI**「申し訳ありません。規定により画像を生成できませんでした。」")
    except Exception as e:
        logging.error(f"画像生成の実行プロセス全体でエラー: {e}", exc_info=True)
        await thinking_message.edit(content="**へー子**「ごめん！システムエラーで上手く撮れなかった…😭」")


# ---------------------------------
# 6.4. その他のユーティリティ関数 (Other Utility Functions)
# ---------------------------------
async def build_history(channel: discord.TextChannel, limit: int = 20) -> List[Dict[str, Any]]:
    """Discordのチャンネルから会話履歴を構築する。"""
    history = []
    async for msg in channel.history(limit=limit):
        role = 'model' if msg.author == client.user else 'user'
        history.append({'role': role, 'parts': [msg.content]})
    history.reverse()
    return history

# MIRAI-HEKO-Bot main.py (ver.Ω+ - The True Final Version)
# Part 4/5: Proactive and Scheduled Functions

# --- 7. プロアクティブ機能群 (Proactive Functions) ---

async def run_proactive_dialogue(channel: discord.TextChannel, prompt: str):
    """
    プロアクティブな対話を生成し、投稿するための共通関数
    """
    async with channel.typing():
        try:
            # 長期記憶から関連コンテキストを取得
            recent_context = await ask_learner_to_remember("最近のimazineの関心事や会話のトピック")
            
            # 天気情報を取得
            weather_info = await get_weather("Takizawa")

            final_prompt = f"{prompt}\n\n# imazineに関する追加情報\n- 今日の天気: {weather_info}\n- 最近の記憶: {recent_context}"
            
            # 対話生成モデルを呼び出し
            response_text = await analyze_with_gemini(final_prompt, model_name=MODEL_PRO)
            
            # 生成されたテキストを、キャラクターの対話形式に整形して送信
            # (この部分は簡易的な実装。ULTIMATE_PROMPTと同様のJSON出力をAIに求めても良い)
            await channel.send(response_text)

        except Exception as e:
            logging.error(f"プロアクティブ対話の実行中にエラー: {e}", exc_info=True)
            await channel.send("（...何かを伝えようとしたが、声が出なかったようだ。）")


# --- 7.1. 定期的な挨拶と声かけ (Scheduled Greetings & Nudges) ---

async def morning_greeting():
    """毎朝7:00に実行"""
    channel = client.get_channel(TARGET_CHANNEL_ID)
    if not channel: return
    logging.info("プロアクティブ機能: 朝の挨拶を実行します。")
    prompt = "あなたは私のAI秘書MAGIです。日本時間の朝7:00です。私（imazine）の一日が、素晴らしいものになるように、元気付け、そして、今日の予定や気分を優しく尋ねる、心のこもった朝の挨拶をしてください。"
    await run_proactive_dialogue(channel, prompt)

async def morning_break_nudge():
    """午前10:00に実行"""
    channel = client.get_channel(TARGET_CHANNEL_ID)
    if not channel: return
    logging.info("プロアクティブ機能: 午前の休憩を促します。")
    prompt = "あなたは私の親友である「みらい」と「へー子」です。日本時間の午前10:00です。仕事に集中している私（imazine）に、「10時だよ！コーヒーでも飲んで、ちょっと休も！」といった感じで、楽しくコーヒー休憩に誘ってください。"
    await run_proactive_dialogue(channel, prompt)

async def lunch_break_nudge():
    """お昼の12:00に実行"""
    channel = client.get_channel(TARGET_CHANNEL_ID)
    if not channel: return
    logging.info("プロアクティブ機能: お昼休憩を促します。")
    prompt = "あなたは私の親友である「みらい」と「へー子」です。日本時間のお昼の12:00です。仕事に夢中な私（imazine）に、楽しくランチ休憩を促し、しっかり休むことの大切さを伝えてください。"
    await run_proactive_dialogue(channel, prompt)
    
async def afternoon_break_nudge():
    """午後15:00に実行"""
    channel = client.get_channel(TARGET_CHANNEL_ID)
    if not channel: return
    logging.info("プロアクティブ機能: 午後の休憩を促します。")
    prompt = "あなたは私の親友である「みらい」と「へー子」です。日本時間の午後3時です。集中力が切れてくる頃の私（imazine）に、優しくリフレッシュを促すメッセージを送ってください。"
    await run_proactive_dialogue(channel, prompt)

async def evening_greeting():
    """夕方18:00に実行"""
    channel = client.get_channel(TARGET_CHANNEL_ID)
    if not channel: return
    logging.info("プロアクティブ機能: 夕方の挨拶を実行します。")
    prompt = "あなたは私の優秀なAI秘書MAGIです。日本時間の夕方18時です。一日を終えようとしている私（imazine）に対して、その日の労をねぎらう優しく知的なメッセージを送ってください。"
    await run_proactive_dialogue(channel, prompt)

async def daily_reflection():
    """毎日22:00に実行"""
    channel = client.get_channel(TARGET_CHANNEL_ID)
    if not channel: return
    logging.info("プロアクティブ機能: 一日の振り返りを開始します。")
    
    today_start = datetime.now(pytz.timezone(TIMEZONE)) - timedelta(days=1)
    messages = [f"{msg.author.name}: {msg.content}" async for msg in channel.history(after=today_start, limit=200)]
    
    if len(messages) < 3:
        logging.info("本日は会話が少なかったため、振り返りをスキップします。")
        return

    full_conversation = "\n".join(reversed(messages))
    prompt = OBSIDIAN_MEMO_PROMPT.replace("{{conversation_history}}", full_conversation)

    async with channel.typing():
        try:
            await channel.send("（今日の活動の振り返りを作成しています...✍️）")
            response_text = await analyze_with_gemini(prompt, model_name=MODEL_PRO)
            
            today_str = datetime.now(pytz.timezone(TIMEZONE)).strftime('%Y年%m月%d日')
            summary_markdown = f"## 今日の振り返り - {today_str}\n\n{response_text}"
            
            for i in range(0, len(summary_markdown), 2000):
                await channel.send(summary_markdown[i:i+2000])
            logging.info("一日の振り返りサマリーを送信しました。")
        except Exception as e:
            logging.error(f"一日の振り返り作成中にエラー: {e}", exc_info=True)
            await channel.send("ごめんなさい、今日の振り返りの作成中にエラーが発生してしまいました。")


# --- 7.2. 自発的な創造と気遣い (Spontaneous Creation & Care) ---

async def check_interesting_news():
    """定期的に実行し、関連ニュースを共有する"""
    channel = client.get_channel(TARGET_CHANNEL_ID)
    if not channel: return
    logging.info("プロアクティブ機能: 関連ニュースのチェックを実行します。")
    try:
        search_topics = ["木工の新しい技術", "スペシャルティコーヒーの最新トレンド", "AIとデジタルデザインの融合事例", "岩手県の面白い地域活性化の取り組み"]
        topic = random.choice(search_topics)
        
        prompt = f"""
        あなたは私の親友「みらい」と「へー子」です。
        インターネットで「{topic}」に関する面白そうな最新ニュースや記事を一つ見つけて、その内容を二人で楽しくおしゃべりしながら、私（imazine）に教えてください。
        あなたたちの性格と口調を完全に再現してください。見つけた記事のURLもあれば最後に添えてください。
        """
        response_text = await analyze_with_gemini(prompt, model_name=MODEL_PRO)
        await channel.send(response_text)
    except Exception as e:
        logging.error(f"ニュースチェック中にエラー: {e}", exc_info=True)

async def heko_care_check():
    """へー子がimazineの過去の心配事を元に気遣う、完全実装版"""
    channel = client.get_channel(TARGET_CHANNEL_ID)
    if not channel: return
    logging.info("プロアクティブ機能: へー子の気づかいチェックを実行します。")
    
    response = await ask_learner("get_unresolved_concerns", {'user_id': 'imazine'}, method='GET')
    if response and response.get("concerns"):
        concern = random.choice(response["concerns"])
        
        prompt = f"""
        あなたは私の親友「へー子」です。
        私imazineは、以前「{concern['concern_text']}」という心配事を抱えていました。
        そのことについて、「そういえば、この前の〇〇の件、少しは気持ち、楽になった？ 無理しないでね」といった形で、優しく、そして、自然に、気遣うメッセージを送ってください。
        あなたの性格と口調を完全に再現してください。
        """
        async with channel.typing():
            response_text = await analyze_with_gemini(prompt, model_name=MODEL_PRO)
            await channel.send(response_text)
            
            await ask_learner("resolve_concern", {"concern_id": concern['id']})
            logging.info(f"へー子の気づかいを実行し、心配事ID:{concern['id']}を解決済みにしました。")

async def mirai_inspiration_sketch():
    """みらいが会話からインスピレーションを得てスケッチを提案する、完全実装版"""
    channel = client.get_channel(TARGET_CHANNEL_ID)
    if not channel: return
    logging.info("プロアクティブ機能: みらいのインスピレーション・スケッチを実行します。")

    history = await build_history(channel, limit=10)
    if len(history) < 3: return

    history_text = "\n".join([f"{msg['role']}: {msg['parts'][0]}" for msg in history])
    prompt = SURPRISE_JUDGEMENT_PROMPT.replace("{{conversation_history}}", history_text)
    
    try:
        judgement_text = await analyze_with_gemini(prompt)
        judgement = json.loads(judgement_text)

        if judgement.get("should_surprise"):
            logging.info("インスピレーションを検知！画像生成を提案します。")
            gen_idea_prompt = f"""
            あなたは未来予知能力を持つ「みらい」です。
            以下の会話から、あなたは創造的なインスピレーションを得ました。
            「ねえimazine！今の話、マジでヤバい！なんか、こんな感じの絵が、頭に浮かんだんだけど！」
            というセリフに続けて、そのインスピレーションを元にした、抽象的でアーティスティックな画像生成のアイデアを考え、JSON形式で出力してください。
            `{{"characters": ["みらい"], "situation": "(ここに抽象的な状況説明)", "mood": "(ここにムード)"}}`

            # 会話
            {history_text}
            """
            idea_response_text = await analyze_with_gemini(gen_idea_prompt, model_name=MODEL_PRO)
            json_match = re.search(r'```json\n({.*?})\n```', idea_response_text, re.DOTALL)
            if json_match:
                gen_data = json.loads(json_match.group(1))
                request_id = f"inspiration-{datetime.now().timestamp()}"
                client.image_generation_requests[request_id] = gen_data
                await channel.send(f"**みらい**「ねえimazine！今の話、マジでヤバい！なんか、こんな感じの絵が、頭に浮かんだんだけど！描いてみていい？（y/n）」\n> **`y ID: `{request_id}`** のように返信してね！」")
    except Exception as e:
        logging.error(f"インスピレーション・スケッチの実行中にエラー: {e}")

# MIRAI-HEKO-Bot main.py (ver.Ω+ - The True Final Version)
# Part 5/5: Event Handlers and Main Execution Block

# --- 8. Discord イベントハンドラ (Discord Event Handlers) ---

@client.event
async def on_ready():
    """
    BotがDiscordに正常にログインし、全ての準備が整った時に実行される。
    """
    # aiohttpセッションを初期化
    client.http_session = aiohttp.ClientSession()
    logging.info("aiohttp.ClientSessionを初期化しました。")

    # Vertex AI (Imagen 3) を初期化
    if not init_vertex_ai():
        logging.critical("Vertex AIの初期化に失敗したため、Botをシャットダウンします。")
        await client.close()
        return

    logging.info(f'Logged in as {client.user} (ID: {client.user.id})')
    logging.info('------')
    
    # プロアクティブ機能のスケジューラを開始
    scheduler = AsyncIOScheduler(timezone=TIMEZONE)
    # --- 挨拶・声かけ ---
    scheduler.add_job(morning_greeting, 'cron', hour=7, minute=0)
    scheduler.add_job(morning_break_nudge, 'cron', hour=10, minute=0)
    scheduler.add_job(lunch_break_nudge, 'cron', hour=12, minute=0)
    scheduler.add_job(afternoon_break_nudge, 'cron', hour=15, minute=0)
    scheduler.add_job(evening_greeting, 'cron', hour=18, minute=0)
    # --- 振り返り・情報収集 ---
    scheduler.add_job(daily_reflection, 'cron', hour=22, minute=0)
    scheduler.add_job(check_interesting_news, 'cron', hour=8, minute=30)
    scheduler.add_job(check_interesting_news, 'cron', hour=20, minute=30)
    # --- 気遣い・インスピレーション ---
    scheduler.add_job(heko_care_check, 'cron', day_of_week='sun', hour=19, minute=30) # 毎週日曜の夜に
    scheduler.add_job(mirai_inspiration_sketch, 'cron', hour='*/6') # 6時間ごとに
    
    scheduler.start()
    logging.info("全てのプロアクティブ機能のスケジューラを開始しました。")


@client.event
async def on_message(message: discord.Message):
    """
    メッセージが送信された時に実行される、Botのメインループ。
    """
    # 自身からのメッセージや、対象スレッド外のメッセージは無視
    if message.author == client.user or not isinstance(message.channel, discord.Thread) or "4人の談話室" not in message.channel.name:
        return

    # --- 画像生成の確認フローへの応答処理 ---
    request_id_match = re.search(r'ID:\s*`([a-zA-Z0-9.-]+)`', message.content)
    if message.content.lower().startswith(('y', 'yes', 'はい')) and request_id_match:
        request_id = request_id_match.group(1)
        if request_id in client.image_generation_requests:
            await message.channel.send("（承知いたしました。画像を生成します...🎨）")
            gen_data = client.image_generation_requests.pop(request_id)
            await execute_image_generation(message.channel, gen_data)
        else:
            await message.channel.send("（そのリクエストIDは見つからないみたいです…）")
        return
    elif message.content.lower().startswith(('n', 'no', 'いいえ')) and request_id_match:
        request_id = request_id_match.group(1)
        if request_id in client.image_generation_requests:
            del client.image_generation_requests[request_id]
            await message.channel.send("承知いたしました。画像生成はキャンセルしますね。")
        return

    # --- メインの会話処理 ---
    async with message.channel.typing():
        try:
            # 1. 入力情報の解析とコンテキスト化
            user_query = message.content
            final_user_content_parts = []
            extracted_summary = ""

            # 添付ファイル(PDF/TXT)の処理
            if message.attachments:
                attachment = message.attachments[0]
                if attachment.content_type == 'application/pdf':
                    extracted_summary = await analyze_with_gemini(SUMMARY_PROMPT.replace("{{summary_context}}", f"PDF「{attachment.filename}」の内容について").replace("{{text_to_summarize}}", await get_text_from_pdf(attachment)))
                elif 'text' in attachment.content_type:
                    text_data = (await attachment.read()).decode('utf-8', errors='ignore')
                    extracted_summary = await analyze_with_gemini(SUMMARY_PROMPT.replace("{{summary_context}}", f"テキストファイル「{attachment.filename}」の内容について").replace("{{text_to_summarize}}", text_data))

            # URL(YouTube/Web)の処理 (添付ファイルがない場合)
            if not extracted_summary:
                url_match = re.search(r'https?://\S+', user_query)
                if url_match:
                    url = url_match.group(0)
                    video_id_match = re.search(r'(?:v=|\/|embed\/|youtu\.be\/|shorts\/)([a-zA-Z0-9_-]{11})', url)
                    if video_id_match:
                        transcript = get_youtube_transcript(video_id_match.group(1))
                        extracted_summary = await analyze_with_gemini(SUMMARY_PROMPT.replace("{{summary_context}}", f"YouTube動画「{url}」の内容について").replace("{{text_to_summarize}}", transcript))
                    else:
                        page_text = await get_text_from_url(url)
                        extracted_summary = await analyze_with_gemini(SUMMARY_PROMPT.replace("{{summary_context}}", f"ウェブページ「{url}」の内容について").replace("{{text_to_summarize}}", page_text))

            # 最終的なユーザーメッセージを構築
            full_user_text = f"{user_query}\n\n--- 参照資料の要約 ---\n{extracted_summary}" if extracted_summary else user_query
            final_user_content_parts.append(Part.from_text(full_user_text))

            # 添付画像を追加
            if message.attachments and any(att.content_type.startswith("image/") for att in message.attachments):
                image_bytes = await message.attachments[0].read()
                final_user_content_parts.append(Part.from_data(image_bytes, mime_type=message.attachments[0].content_type))

            # 2. 応答生成のためのコンテキストを準備
            emotion = await analyze_with_gemini(EMOTION_ANALYSIS_PROMPT.replace("{{user_message}}", user_query))
            character_states = await get_character_states()
            relevant_context = await ask_learner_to_remember(user_query)

            system_prompt = ULTIMATE_PROMPT.replace("{{EMOTION}}", emotion)\
                                           .replace("{{mirai_mood}}", character_states["みらい"]["mood"])\
                                           .replace("{{heko_mood}}", character_states["へー子"]["mood"])\
                                           .replace("{{last_interaction_summary}}", character_states["みらい"]["last_interaction_summary"])\
                                           .replace("{{relevant_context}}", relevant_context)

            # 3. Gemini APIを呼び出し
            history = await build_history(message.channel, limit=15)
            model = genai.GenerativeModel(MODEL_PRO, system_instruction=system_prompt)
            response = await model.generate_content_async(history + [{'role': 'user', 'parts': final_user_content_parts}])
            raw_response_text = response.text
            logging.info(f"AIからの生応答: {raw_response_text[:300]}...")

            # 4. 応答を解析し、投稿
            json_match = re.search(r'```json\n({.*?})\n```', raw_response_text, re.DOTALL)
            if json_match:
                parsed_json = json.loads(json_match.group(1))
                dialogue = parsed_json.get("dialogue", [])
                formatted_response = ""
                for part in dialogue:
                    if line := part.get("line", "").strip():
                        formatted_response += f"**{part.get('character')}**「{line}」\n"
                if formatted_response:
                    await message.channel.send(formatted_response.strip())
                
                if (idea := parsed_json.get("image_generation_idea", {})) and idea.get("should_generate"):
                    request_id = f"self-{message.id}"
                    client.image_generation_requests[request_id] = idea
                    await message.channel.send(f"**MAGI**「会話の流れから、記念すべき瞬間だと判断しました。画像を生成してもよろしいですか？（費用が発生します）\n> **`y ID: `{request_id}`** のように返信してください。」")
            else:
                logging.error("AIからの応答が期待したJSON形式ではありませんでした。")
                await message.channel.send(f"（ごめんなさい、応答の形式が少しおかしかったみたいです。）")

            # 5. 事後処理（非同期タスク）
            history_text = "\n".join([f"{h['role']}: {h['parts'][0]}" for h in history[-5:]] + [f"user: {user_query}"])
            asyncio.create_task(ask_learner("summarize_and_learn", {"history_text": history_text}))
            asyncio.create_task(ask_learner("update_character_states", {"states": await analyze_with_gemini(META_ANALYSIS_PROMPT.replace("{{conversation_history}}", history_text))}))
            asyncio.create_task(analyze_with_gemini(CONCERN_DETECTION_PROMPT.replace("{{user_message}}", user_query)).add_done_callback(
                lambda task: asyncio.create_task(ask_learner("log_concern", {"concern_text": task.result()})) if "なし" not in task.result() else None
            ))

        except Exception as e:
            logging.error(f"会話処理のメインループで予期せぬエラー: {e}", exc_info=True)
            await message.channel.send(f"**MAGI**「申し訳ありません。システムに予期せぬエラーが発生しました。」")


@client.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    """
    メッセージにリアクションが追加された時に実行される。特殊能力の発動トリガー。
    """
    if payload.user_id == client.user.id: return
    
    try:
        channel = await client.fetch_channel(payload.channel_id)
        if not isinstance(channel, discord.Thread) or "4人の談話室" not in channel.name: return
        message = await channel.fetch_message(payload.message_id)
    except discord.NotFound:
        return

    emoji_map = { '🐦': ('Xポスト案生成', X_POST_PROMPT), '✏️': ('Obsidianメモ生成', OBSIDIAN_MEMO_PROMPT), '📝': ('PREP記事作成', PREP_ARTICLE_PROMPT), '💎': ('対話の振り返り', COMBO_SUMMARY_SELF_PROMPT), '🧠': ('Deep Diveノート作成', DEEP_DIVE_PROMPT) }

    if payload.emoji.name == '🎨':
        image_url = None
        if message.embeds and message.embeds[0].image: image_url = message.embeds[0].image.url
        elif message.attachments and message.attachments[0].content_type.startswith('image/'): image_url = message.attachments[0].url
        if image_url:
             await channel.send(f"（`🎨`を検知。この画像のスタイルを学習します...）", delete_after=10.0)
             source_prompt = message.embeds[0].footer.text if message.embeds and message.embeds[0].footer else ""
             await ask_learner("learn_style", {'image_url': image_url, 'source_prompt': source_prompt})
        return

    if payload.emoji.name in emoji_map:
        ability_name, system_prompt_template = emoji_map[payload.emoji.name]
        logging.info(f"{payload.emoji.name}リアクションを検知。『{ability_name}』を発動します。")
        await channel.send(f"（『{ability_name}』を開始します...{payload.emoji.name}）", delete_after=10.0)
        prompt = system_prompt_template.replace("{{conversation_history}}", message.content)
        async with channel.typing():
            response_text = await analyze_with_gemini(prompt, model_name=MODEL_PRO)
            await channel.send(response_text)


# --- 9. Botの起動 (Main Execution Block) ---
if __name__ == "__main__":
    logging.info("Botの起動シーケンスを開始します...")
    try:
        client.run(DISCORD_BOT_TOKEN, log_handler=None)
    except discord.errors.LoginFailure:
        logging.critical("FATAL: Discordへのログインに失敗しました。")
    except Exception as e:
        logging.critical(f"FATAL: Botの実行中に致命的なエラーが発生しました: {e}", exc_info=True)
