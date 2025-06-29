# learner_main.py (ver.Ω++ - The True Final Version)
# The Temple of Soul and Memory, running on Supabase.
# Part 1/2: Core Setup and Memory I/O

import os
import logging
import datetime as dt
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# LangChain and Vector Store related imports
from langchain_community.vectorstores.supabase import SupabaseVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Supabase Client
from supabase.client import Client, create_client
import google.generativeai as genai

# --- 1. 初期設定 (Initial Setup) ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
app = FastAPI(
    title="Learner API - The Soul of MIRAI-HEKO-Bot",
    description="This API manages the long-term memory, style palette, character states, and soul records.",
    version="3.0.0"
)

# --- 2. クライアントの初期化 (Client Initialization) ---

try:
    # Supabaseクライアントを環境変数から初期化
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not supabase_key:
        raise ValueError("SupabaseのURLまたはサービスロールキーが設定されていません。")
    supabase: Client = create_client(supabase_url, supabase_key)
    logging.info("Supabase client initialized successfully.")

    # GoogleのAPIキーを環境変数から取得
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEYが設定されていません。")
    # genaiクライアントも設定
    genai.configure(api_key=google_api_key)

    # Embeddingモデルを初期化 (知識のベクトル化に使用)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    logging.info("Google Generative AI Embeddings initialized successfully.")

except Exception as e:
    logging.critical(f"FATAL: クライアントの初期化中にエラーが発生しました: {e}")
    raise e

# --- 3. テキスト分割とベクトルストアの設定 (Text Splitter and Vector Store Configuration) ---

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len,
    add_start_index=True,
)

# SupabaseVectorStoreをベクトルストア（記憶の大図書館）として利用
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents" # SQLで作成したRPC関数名
)
logging.info(f"Supabase Vector Store initialized for table 'documents'.")


# --- 4. APIリクエスト/レスポンスモデル定義 (Pydantic Models) ---

class LearnRequest(BaseModel):
    text_content: str = Field(..., description="学習させたいテキスト本文。")
    metadata: Dict[str, Any] = Field({}, description="テキストに関連するメタデータ。")

class LearnResponse(BaseModel):
    status: str = "success"
    message: str
    document_ids: List[str]

class QueryRequest(BaseModel):
    query_text: str = Field(..., description="記憶を検索するための問い合わせテキスト。")
    match_threshold: float = Field(0.7, description="類似度の閾値。")
    match_count: int = Field(5, description="取得する最大チャンク数。")

class QueryResponse(BaseModel):
    status: str = "success"
    documents: List[str] = Field(..., description="検索クエリに最も関連性の高い記憶の断片リスト。")


# --- 5. 基本的な記憶の読み書きAPI (Core Memory Endpoints) ---

@app.post("/learn", response_model=LearnResponse, tags=["Memory"])
async def learn_document(request: LearnRequest):
    """
    新しい知識（テキスト）を学習し、ベクトル化して`documents`テーブルに保管する。
    また、学習した記録を`learning_history`テーブルに保存する。
    """
    if not request.text_content.strip():
        raise HTTPException(status_code=400, detail="学習するテキスト内容が空です。")
    try:
        logging.info(f"新しい知識の学習を開始します。ソース: {request.metadata.get('filename', 'Unknown')}")
        
        # テキストをチャンクに分割
        docs = text_splitter.create_documents([request.text_content], metadatas=[request.metadata])
        
        # ベクトルストアにチャンクを追加
        doc_ids = vector_store.add_documents(docs, returning="minimal")
        logging.info(f"{len(docs)}個のチャンクをベクトルストアに正常に追加しました。")

        # Supabaseの`learning_history`テーブルにも学習履歴を記録
        try:
            history_record = {
                "user_id": request.metadata.get("user_id"),
                "username": request.metadata.get("username"),
                "filename": request.metadata.get("filename"),
                "file_size": request.metadata.get("file_size")
            }
            supabase.table('learning_history').insert(history_record).execute()
            logging.info("Supabaseの`learning_history`への記録に成功しました。")
        except Exception as db_error:
            logging.warning(f"Supabaseの`learning_history`への記録中に警告: {db_error}")

        return LearnResponse(
            message="Knowledge successfully acquired and stored in the vector temple.",
            document_ids=doc_ids
        )
    except Exception as e:
        logging.error(f"学習処理(/learn)中にエラーが発生しました: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/query", response_model=QueryResponse, tags=["Memory"])
async def query_memory(request: QueryRequest):
    """
    問い合わせ内容に基づいて、`documents`テーブルから最も関連性の高い記憶を検索して返す。
    """
    if not request.query_text.strip():
        raise HTTPException(status_code=400, detail="検索クエリが空です。")
    try:
        logging.info(f"記憶の検索を実行します。クエリ: 「{request.query_text}」")
        
        # 類似度検索を実行
        docs = vector_store.similarity_search(
            query=request.query_text,
            k=request.match_count
        )
        
        response_docs = [doc.page_content for doc in docs]
        logging.info(f"問い合わせに対して{len(response_docs)}件の関連する記憶を返却します。")
        
        return QueryResponse(documents=response_docs)
    except Exception as e:
        logging.error(f"記憶の検索(/query)中にエラーが発生しました: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/", tags=["System"])
async def root():
    """
    APIサーバーの生存確認用エンドポイント。
    """
    return {"message": "Learner is awake. The soul of imazine's world is waiting for a command."}

# learner_main.py (ver.Ω++ - The True Final Version)
# Part 2/2: Advanced Functions for Style, Emotion, Soul, and Growth

# --- 6. 高度な機能のAPIモデル定義 (Pydantic Models for Advanced Features) ---

class StyleLearnRequest(BaseModel):
    image_url: str = Field(..., description="学習させたい画風の画像URL。")
    source_prompt: Optional[str] = Field("", description="その画像が生成された際の、元のプロンプト（もしあれば）。")

class StyleLearnResponse(BaseModel):
    status: str = "success"
    message: str
    style_id: int

class CharacterState(BaseModel):
    mirai_mood: str
    heko_mood: str
    last_interaction_summary: str

class Concern(BaseModel):
    user_id: str = "imazine"
    concern_text: str

class ResolveConcernRequest(BaseModel):
    concern_id: int

class MagiSoulSyncRequest(BaseModel):
    learned_from_filename: str
    soul_record: str


# --- 7. 画風・感情・成長を司るAPI (Endpoints for Style, Emotion, and Growth) ---

@app.post("/styles", tags=["Style Palette"])
async def learn_style(request: StyleLearnRequest):
    """
    imazineが🎨リアクションを付けた画像のスタイルを学習し、`styles`に保存する。
    このバージョンでは、まずGeminiに画像とプロンプトを渡し、分析させる。
    """
    try:
        logging.info(f"新しい画風の学習を開始します。ソースURL: {request.image_url}")
        
        # Gemini Visionに画像とプロンプトを渡して、スタイルを分析させる
        vision_prompt = f"""
        あなたは世界クラスの美術評論家です。
        添付された画像は、以下のプロンプトを元にAIによって生成されました。
        - 元プロンプト: {request.source_prompt if request.source_prompt else "なし"}
        
        この画像の芸術的なスタイルを、以下の観点から詳細に分析し、その結果を厳密なJSON形式で出力してください。
        - 色彩（Color Palette）
        - 光と影（Lighting & Shadow）
        - 構図（Composition）
        - 全体的な雰囲気（Overall Mood）
        - 特徴的なキーワード（5個程度）
        """
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        image_response = await model.generate_content_async([vision_prompt, {"mime_type": "image/jpeg", "data": requests.get(request.image_url).content}])
        
        style_analysis_json = json.loads(re.search(r'```json\n({.*?})\n```', image_response.text, re.DOTALL).group(1))

        response = supabase.table('styles').insert({
            "source_prompt": request.source_prompt,
            "source_image_url": request.image_url,
            "style_analysis_json": style_analysis_json
        }).execute()

        style_id = response.data[0]['id']
        logging.info(f"新しい画風をID:{style_id}として`styles`に記録しました。")
        return {"status": "success", "message": "Style analyzed and learned.", "style_id": style_id}
    except Exception as e:
        logging.error(f"スタイル学習(/styles)中にエラーが発生しました: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/styles", tags=["Style Palette"])
async def get_styles():
    """現在学習済みの画風（スタイル）の分析結果リストを取得する。"""
    try:
        res = supabase.table('styles').select("style_analysis_json").order('created_at', desc=True).limit(5).execute()
        return {"documents": [item['style_analysis_json'] for item in res.data]}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/character_state", tags=["Character Emotion"])
async def update_character_state(request: CharacterState):
    """会話の終了後、Botから送られてきたキャラクターの最新の感情状態でDBを更新する。"""
    try:
        # 常に最新の1行を更新するため、既存のデータを削除してから新しいデータを挿入する
        supabase.table('character_states').delete().neq('id', 0).execute() # 全削除
        supabase.table('character_states').insert(request.model_dump()).execute()
        return {"status": "success", "message": "Character state updated."}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/character_state", tags=["Character Emotion"])
async def get_character_state():
    """会話の開始前、Botがキャラクターの現在の感情状態をDBから取得する。"""
    try:
        res = supabase.table('character_states').select("*").order('created_at', desc=True).limit(1).execute()
        if res.data:
            return {"status": "success", "state": res.data[0]}
        else: # データがまだない場合
            default_state = {"mirai_mood": "ニュートラル", "heko_mood": "ニュートラル", "last_interaction_summary": "まだ会話が始まっていません。"}
            return {"status": "success", "state": default_state}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/concern", tags=["Character Care"])
async def log_concern(request: Concern):
    try:
        res = supabase.table('concerns').insert({"user_id": request.user_id, "concern_text": request.concern_text}).execute()
        return {"status": "success", "concern_id": res.data[0]['id']}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/unresolved_concerns", tags=["Character Care"])
async def get_unresolved_concerns(user_id: str = "imazine"):
    try:
        res = supabase.table('concerns').select("*").eq('user_id', user_id).is_('notified_at', 'null').order('created_at').limit(5).execute()
        return {"concerns": res.data}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/resolve_concern", tags=["Character Care"])
async def mark_concern_notified(request: ResolveConcernRequest):
    try:
        supabase.table('concerns').update({"notified_at": dt.datetime.now(dt.timezone.utc).isoformat()}).eq('id', request.concern_id).execute()
        return {"status": "success"}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/gals_vocabulary", tags=["Vocabulary"])
async def get_gals_vocabulary():
    """みらいとへー子が使いそうな単語を取得する"""
    try:
        res = supabase.table('gals_vocabulary').select("word, character_type").limit(20).execute()
        return {"vocabulary": res.data}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/magi_soul", tags=["Magi's Soul"])
async def sync_magi_soul(request: MagiSoulSyncRequest):
    """Geminiとの対話の記録を、MAGIの魂として蓄積する"""
    try:
        res = supabase.table('magi_soul').insert({
            "learned_from_filename": request.learned_from_filename,
            "soul_record": request.soul_record
        }).execute()
        return {"status": "success", "record_id": res.data[0]['id']}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/magi_soul", tags=["Magi's Soul"])
async def get_latest_magi_soul():
    """MAGIの人格に反映させるため、最新の魂の記録を取得する"""
    try:
        # 最新の3つの対話記録を取得し、結合して返す
        res = supabase.table('magi_soul').select("soul_record").order('created_at', desc=True).limit(3).execute()
        records = [item['soul_record'] for item in res.data]
        return {"soul_record": "\n---\n".join(records)}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
