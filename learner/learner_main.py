# learner_main.py (ver.Ω++ - The True Final Version)
# Creator & Partner: imazine & Gemini
# Part 1/2: Core Setup and Memory I/O

import os
import logging
import datetime as dt
from fastapi import FastAPI, HTTPException, Request
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
app = FastAPI(title="Learner API - The Soul of MIRAI-HEKO-Bot", version="4.0.0")

# --- 2. クライアント初期化 ---
try:
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not supabase_key: raise ValueError("SupabaseのURL/キーが設定されていません。")
    supabase: Client = create_client(supabase_url, supabase_key)
    
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key: raise ValueError("GOOGLE_API_KEYが設定されていません。")
    genai.configure(api_key=google_api_key)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    vector_store = SupabaseVectorStore(client=supabase, embedding=embeddings, table_name="documents", query_name="match_documents")
except Exception as e:
    logging.critical(f"FATAL: 初期化中にエラー: {e}")
    raise e

# --- 3. Pydanticモデル定義 ---
class LearnRequest(BaseModel): text_content: str; metadata: Dict[str, Any] = {}
class QueryRequest(BaseModel): query_text: str
class StyleLearnRequest(BaseModel): image_url: str; source_prompt: Optional[str] = ""
class CharacterState(BaseModel): mirai_mood: str; heko_mood: str; last_interaction_summary: str
class Concern(BaseModel): user_id: str = "imazine"; concern_text: str
class ResolveConcernRequest(BaseModel): concern_id: int
class MagiSoulSyncRequest(BaseModel): learned_from_filename: str; soul_record: str

# --- 4. APIエンドポイント (基本機能) ---

@app.post("/learn", tags=["Memory"])
async def learn_document(request: LearnRequest):
    try:
        docs = text_splitter.create_documents([request.text_content], metadatas=[request.metadata])
        vector_store.add_documents(docs)
        
        history_record = {
            "user_id": request.metadata.get("user_id"), "username": request.metadata.get("username"),
            "filename": request.metadata.get("filename"), "file_size": request.metadata.get("file_size")
        }
        supabase.table('learning_history').insert(history_record).execute()
        return {"status": "success", "message": "Knowledge and history acquired."}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", tags=["Memory"])
async def query_memory(request: QueryRequest):
    try:
        docs = vector_store.similarity_search(request.query_text, k=5)
        return {"documents": [doc.page_content for doc in docs]}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/", tags=["System"])
async def root(): return {"message": "Learner is awake. The soul of imazine's world is waiting."}

# learner_main.py (ver.Ω++ - The True Final Version)
# Part 2/2: Advanced Functions for Style, Emotion, Soul, and Growth

# --- 5. APIエンドポイント (高度な機能) ---

STYLE_ANALYSIS_PROMPT = """
あなたは世界クラスの美術評論家です。添付された画像と、それが生成されたプロンプト（あれば）を元に、この画像の芸術的スタイルを詳細に分析し、結果を厳密なJSON形式で出力してください。
- 元プロンプト: {{source_prompt}}
- 分析項目: 色彩(Color Palette), 光と影(Lighting & Shadow), 構図(Composition), 全体的な雰囲気(Overall Mood), 特徴的なキーワード(5個)
"""

@app.post("/styles", tags=["Style Palette"])
async def analyze_and_learn_style(request: StyleLearnRequest):
    """画像からスタイルを分析し、`styles`テーブルに保存する"""
    try:
        logging.info(f"新しい画風の学習を開始します。ソースURL: {request.image_url}")
        
        model = genai.GenerativeModel('gemini-2.5-pro-preview-03-25')
        
        # URLから画像データを取得
        image_response = requests.get(request.image_url)
        image_response.raise_for_status()
        image_content = image_response.content

        prompt = STYLE_ANALYSIS_PROMPT.replace("{{source_prompt}}", request.source_prompt if request.source_prompt else "なし")
        
        # Geminiに画像とプロンプトを渡して分析させる
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
        
        return {"status": "success", "style_id": res.data[0]['id']}
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
        res = supabase.table('concerns').insert({"user_id": request.user_id, "concern_text": request.concern_text}).execute()
        return {"status": "success", "concern_id": res.data[0]['id']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/unresolved_concerns", tags=["Character Care"])
async def get_unresolved_concerns(user_id: str = "imazine"):
    try:
        res = supabase.table('concerns').select("*").eq('user_id', user_id).is_('notified_at', 'null').order('created_at').limit(5).execute()
        return {"concerns": res.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/resolve_concern", tags=["Character Care"])
async def mark_concern_notified(request: ResolveConcernRequest):
    try:
        supabase.table('concerns').update({"notified_at": dt.datetime.now(dt.timezone.utc).isoformat()}).eq('id', request.concern_id).execute()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gals_vocabulary", tags=["Vocabulary"])
async def get_gals_vocabulary():
    """みらいとへー子が使いそうな単語を取得する"""
    try:
        res = supabase.table('gals_vocabulary').select("word, character_type").limit(30).execute()
        return {"vocabulary": res.data}
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
