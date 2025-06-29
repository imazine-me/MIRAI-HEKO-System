# learner_main.py (ver.Ω++ - The True Final Version)
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
app = FastAPI(title="Learner API - The Soul of MIRAI-HEKO-Bot", version="4.0.0")

# --- 2. クライアント初期化 ---
try:
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    supabase: Client = create_client(supabase_url, supabase_key)
    
    google_api_key = os.environ.get("GOOGLE_API_KEY")
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

@app.post("/styles", tags=["Style Palette"])
async def analyze_and_learn_style(request: StyleLearnRequest):
    """画像からスタイルを分析し、`styles`テーブルに保存する"""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        image_content = requests.get(request.image_url).content
        
        # あなたが提供してくださったSTYLE_ANALYSIS_PROMPT
        prompt = STYLE_ANALYSIS_PROMPT.replace("{{original_prompt}}", request.source_prompt if request.source_prompt else "なし")
        response = await model.generate_content_async([prompt, {"mime_type": "image/jpeg", "data": image_content}])
        
        json_match = re.search(r'```json\n({.*?})\n```', response.text, re.DOTALL)
        if not json_match: raise ValueError("Style analysis did not return valid JSON.")
        style_analysis_json = json.loads(json_match.group(1))

        insert_data = {
            "source_prompt": request.source_prompt,
            "source_image_url": request.image_url,
            "style_analysis_json": style_analysis_json,
            "style_name": style_analysis_json.get("style_name", "Untitled Style")
        }
        res = supabase.table('styles').insert(insert_data).execute()
        return {"status": "success", "style_id": res.data[0]['id']}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/styles", tags=["Style Palette"])
async def get_styles():
    """現在学習済みの画風（スタイル）の分析結果リストを取得する"""
    try:
        res = supabase.table('styles').select("style_analysis_json").order('created_at', desc=True).limit(5).execute()
        return {"styles": [item['style_analysis_json'] for item in res.data]}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/character_state", tags=["Character Emotion"])
async def update_character_state(request: CharacterState):
    """キャラクターの最新の感情状態でDBを更新する"""
    try:
        supabase.table('character_states').delete().neq('id', 0).execute() # 常に1行だけ保持
        supabase.table('character_states').insert(request.model_dump()).execute()
        return {"status": "success"}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/character_state", tags=["Character Emotion"])
async def get_character_state():
    """キャラクターの現在の感情状態をDBから取得する"""
    try:
        res = supabase.table('character_states').select("*").order('id', desc=True).limit(1).execute()
        if res.data:
            return {"state": res.data[0]}
        else:
            # データがない場合はデフォルト値を返す
            return {"state": {"mirai_mood": "ニュートラル", "heko_mood": "ニュートラル", "last_interaction_summary": "まだ会話が始まっていません。"}}
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
        res = supabase.table('gals_vocabulary').select("word, character_type").limit(30).execute()
        return {"vocabulary": res.data}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/magi_soul", tags=["Magi's Soul"])
async def sync_magi_soul(request: MagiSoulSyncRequest):
    """Geminiとの対話の記録を、MAGIの魂として蓄積する"""
    try:
        res = supabase.table('magi_soul').insert({"learned_from_filename": request.learned_from_filename, "soul_record": request.soul_record}).execute()
        return {"status": "success", "record_id": res.data[0]['id']}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/magi_soul", tags=["Magi's Soul"])
async def get_latest_magi_soul():
    """MAGIの人格に反映させるため、最新の魂の記録を取得する"""
    try:
        res = supabase.table('magi_soul').select("soul_record").order('created_at', desc=True).limit(5).execute()
        records = [item['soul_record'] for item in res.data]
        return {"soul_record": "\n---\n".join(records)}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
