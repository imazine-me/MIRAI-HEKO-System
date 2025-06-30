# learner_main.py (ver.Ω++ - The Final Truth, Rev.3)
# Creator & Partner: imazine & Gemini
# Part 1/2: Core Setup and Memory I/O

import os
import logging
import datetime as dt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from langchain_community.vectorstores.supabase import SupabaseVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from supabase.client import Client, create_client
import requests
import re
import json

# --- 1. 初期設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
app = FastAPI(
    title="Learner API - The Soul of MIRAI-HEKO-Bot",
    description="This API manages the long-term memory, style palette, character states, and soul records, based on imazine's final design.",
    version="4.2.0" # Reflecting the latest fixes
)

# --- 2. クライアント初期化 ---
try:
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not supabase_key:
        raise ValueError("SupabaseのURLまたはサービスロールキーが設定されていません。")
    supabase: Client = create_client(supabase_url, supabase_key)
    logging.info("Supabase client initialized successfully.")

    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEYが設定されていません。")
    genai.configure(api_key=google_api_key)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    logging.info("Google Generative AI Embeddings initialized successfully.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True,
    )

    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents"
    )
    logging.info(f"Supabase Vector Store initialized for table 'documents'.")

except Exception as e:
    logging.critical(f"FATAL: クライアント初期化中にエラー: {e}")
    raise e

# --- 3. Pydanticモデル定義 (基本機能) ---
class LearnRequest(BaseModel):
    text_content: str = Field(..., description="学習させたいテキスト本文。")
    metadata: Dict[str, Any] = Field({}, description="テキストに関連するメタデータ。")

class LearnResponse(BaseModel):
    status: str = "success"
    message: str

class QueryRequest(BaseModel):
    query_text: str = Field(..., description="記憶を検索するための問い合わせテキスト。")

class QueryResponse(BaseModel):
    status: str = "success"
    documents: List[str] = Field(..., description="検索クエリに最も関連性の高い記憶の断片リスト。")


# --- 4. APIエンドポイント (基本機能) ---

@app.post("/learn", response_model=LearnResponse, tags=["Memory"])
async def learn_document(request: LearnRequest):
    """
    新しい知識を学習し、`documents`テーブルにベクトルとして保管する。
    また、学習履歴を`learning_history`テーブルに保存する。
    """
    if not request.text_content.strip():
        raise HTTPException(status_code=400, detail="学習するテキスト内容が空です。")
    try:
        logging.info(f"新しい知識の学習を開始します。ソース: {request.metadata.get('filename', 'Unknown')}")
        
        docs = text_splitter.create_documents([request.text_content], metadatas=[request.metadata])
        vector_store.add_documents(docs)
        logging.info(f"{len(docs)}個のチャンクをベクトルストアに正常に追加しました。")

        # あなたのDB設計に完全に準拠 (user_id)
        history_record = {
            "user_id": request.metadata.get("user_id"),
            "username": request.metadata.get("username"),
            "filename": request.metadata.get("filename"),
            "file_size": request.metadata.get("file_size")
        }
        supabase.table('learning_history').insert(history_record).execute()
        logging.info("Supabaseの`learning_history`への記録に成功しました。")

        return LearnResponse(message="Knowledge successfully acquired and history logged.")
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
        
        docs = vector_store.similarity_search(query=request.query_text, k=5)
        response_docs = [doc.page_content for doc in docs]
        
        logging.info(f"問い合わせに対して{len(response_docs)}件の関連する記憶を返却します。")
        return QueryResponse(status="success", documents=response_docs)
    except Exception as e:
        logging.error(f"記憶の検索(/query)中にエラーが発生しました: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/", tags=["System"])
async def root():
    """APIサーバーの生存確認用エンドポイント。"""
    return {"message": "Learner is awake. The soul of imazine's world is waiting for a command."}

# learner_main.py (ver.Ω++, The Final Truth, Rev.3)
# Part 2/2: Advanced Functions for Style, Emotion, Soul, and Growth

# --- 5. APIリクエスト/レスポンスモデル定義 (高度な機能) ---

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


# --- 6. APIエンドポイント (高度な機能) ---

STYLE_ANALYSIS_PROMPT = """
あなたは世界クラスの美術評論家です。添付された画像と、それが生成されたプロンプト（あれば）を元に、この画像の芸術的スタイルを詳細に分析し、結果を厳密なJSON形式で出力してください。
- 元プロンプト: {{source_prompt}}
- 分析項目: 色彩(Color Palette), 光と影(Lighting & Shadow), 構図(Composition), 全体的な雰囲気(Overall Mood), 特徴的なキーワード(5個)
"""

@app.post("/styles", response_model=StyleLearnResponse, tags=["Style Palette"])
async def analyze_and_learn_style(request: StyleLearnRequest):
    """画像からスタイルを分析し、`styles`テーブルに保存する"""
    try:
        logging.info(f"新しい画風の学習を開始します。ソースURL: {request.image_url}")
        
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        
        image_response = requests.get(request.image_url)
        image_response.raise_for_status()
        image_content = image_response.content

        prompt = STYLE_ANALYSIS_PROMPT.replace("{{source_prompt}}", request.source_prompt if request.source_prompt else "なし")
        
        # 正しい作法でGeminiに画像とプロンプトを渡す
        response = await model.generate_content_async(
            [prompt, {"mime_type": "image/jpeg", "data": image_content}]
        )
        
        json_match = re.search(r'```json\n({.*?})\n```', response.text, re.DOTALL)
        if not json_match:
            raise ValueError("Style analysis did not return valid JSON.")
        style_analysis_json = json.loads(json_match.group(1))

        insert_data = {
            "source_prompt": request.source_prompt,
            "source_image_url": request.image_url,
            "style_analysis_json": style_analysis_json,
            "style_name": style_analysis_json.get("style_name", "Untitled Style")
        }
        res = supabase.table('styles').insert(insert_data).execute()
        
        return StyleLearnResponse(status="success", message="Style analyzed and learned.", style_id=res.data[0]['id'])
    except Exception as e:
        logging.error(f"スタイル学習(/styles)中にエラーが発生しました: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/styles", tags=["Style Palette"])
async def get_styles():
    """現在学習済みの画風（スタイル）の分析結果リストを取得する"""
    try:
        res = supabase.table('styles').select("style_analysis_json").order('created_at', desc=True).limit(5).execute()
        return {"styles": [item['style_analysis_json'] for item in res.data]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/character_state", tags=["Character Emotion"])
async def update_character_state(request: CharacterState):
    """キャラクターの最新の感情状態でDBを更新する"""
    try:
        # 常に最新の1行だけを保持する設計
        supabase.table('character_states').delete().neq('id', 0).execute()
        supabase.table('character_states').insert(request.model_dump()).execute()
        return {"status": "success", "message": "Character state updated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/character_state", tags=["Character Emotion"])
async def get_character_state():
    """キャラクターの現在の感情状態をDBから取得する"""
    try:
        res = supabase.table('character_states').select("*").order('id', desc=True).limit(1).execute()
        if res.data:
            return {"state": res.data[0]}
        else:
            return {"state": {"mirai_mood": "ニュートラル", "heko_mood": "ニュートラル", "last_interaction_summary": "まだ会話が始まっていません。"}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/concern", tags=["Character Care"])
async def log_concern(request: Concern):
    try:
        # あなたのDB設計に完全に準拠 (user_id)
        res = supabase.table('concerns').insert({"user_id": request.user_id, "concern_text": request.concern_text}).execute()
        return {"status": "success", "concern_id": res.data[0]['id']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/unresolved_concerns", tags=["Character Care"])
async def get_unresolved_concerns(user_id: str = "imazine"):
    try:
        # あなたのDB設計に完全に準拠 (notified_at)
        res = supabase.table('concerns').select("*").eq('user_id', user_id).is_('notified_at', 'null').order('created_at').limit(5).execute()
        return {"concerns": res.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/resolve_concern", tags=["Character Care"])
async def mark_concern_notified(request: ResolveConcernRequest):
    try:
        # あなたのDB設計に完全に準拠 (notified_at)
        supabase.table('concerns').update({"notified_at": dt.datetime.now(dt.timezone.utc).isoformat()}).eq('id', request.concern_id).execute()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gals_words", tags=["Vocabulary"])
async def get_gals_words():
    """gals_wordsテーブルから、単語リストを取得する"""
    try:
        # あなたのDB設計に完全に準拠 (gals_words)
        res = supabase.table('gals_words').select("word, character_type").limit(30).execute()
        return {"vocabulary": res.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gals_vocabulary", tags=["Dialogue"])
async def get_gals_vocabulary_examples():
    """gals_vocabularyテーブルから、会話のお手本（語録）を取得する"""
    try:
        # あなたのDB設計に完全に準拠 (gals_vocabulary)
        res = supabase.table('gals_vocabulary').select("example").order('created_at', desc=True).limit(3).execute()
        if res.data:
            examples_text = "\n".join([json.dumps(item['example'], ensure_ascii=False) for item in res.data])
            return {"examples": examples_text}
        return {"examples": "（利用可能な会話例はありません）"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/magi_soul", tags=["Magi's Soul"])
async def sync_magi_soul(request: MagiSoulSyncRequest):
    """Geminiとの対話の記録を、MAGIの魂として蓄積する"""
    try:
        res = supabase.table('magi_soul').insert({
            "learned_from_filename": request.learned_from_filename,
            "soul_record": request.soul_record
        }).execute()
        return {"status": "success", "record_id": res.data[0]['id']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/magi_soul", tags=["Magi's Soul"])
async def get_latest_magi_soul():
    """MAGIの人格に反映させるため、最新の魂の記録を取得する"""
    try:
        res = supabase.table('magi_soul').select("soul_record").order('created_at', desc=True).limit(5).execute()
        records = [item['soul_record'] for item in res.data]
        return {"soul_record": "\n---\n".join(records)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
