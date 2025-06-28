# MIRAI-HEKO-Learner/learner_main.py (ver.9.2 - FastAPI Final)

import os
import logging
import json
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

import google.generativeai as genai
from supabase.client import Client, create_client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from dotenv import load_dotenv

# ローカル開発のために.envファイルから環境変数を読み込む
load_dotenv()

# --- 初期設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- グローバル変数 (FastAPIのライフサイクルで管理) ---
lifespan_context = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- アプリケーション起動時の処理 ---
    logging.info("アプリケーションのライフサイクルが開始します (FastAPI版)...")
    
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')

    if not all([gemini_api_key, supabase_url, supabase_key]):
        raise ValueError("必須の環境変数 (GEMINI_API_KEY, SUPABASE_URL, SUPABASE_KEY) が.envまたは環境変数に設定されていません。")
    
    try:
        lifespan_context["supabase_client"] = create_client(supabase_url, supabase_key)
        logging.info("Supabaseクライアントの初期化に成功。")

        genai.configure(api_key=gemini_api_key)
        lifespan_context["genai_model"] = genai.GenerativeModel('gemini-2.5-pro-preview-03-25')
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
        
        lifespan_context["text_splitter"] = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        
        lifespan_context["vectorstore"] = SupabaseVectorStore(
            client=lifespan_context["supabase_client"],
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents"
        )
        
        logging.info("全ての初期化処理が完了。Learnerは、新しい器(FastAPI)で正常に起動準備完了です。")
        
    except Exception as e:
        logging.critical(f"初期設定中に致命的なエラーが発生しました: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize the application: {e}")
    
    yield
    
    logging.info("アプリケーションのライフサイクルが終了します。")

# --- FastAPIアプリケーションのインスタンス化 ---
app = FastAPI(lifespan=lifespan)

# --- リクエストボディの型定義 ---
class LearnRequest(BaseModel):
    text_content: str

class QueryRequest(BaseModel):
    query_text: str

class SummarizeRequest(BaseModel):
    history_text: str

class StyleRequest(BaseModel):
    style_data: dict

# --- APIエンドポイント ---
@app.get("/")
async def index():
    return {"message": "MIRAI-HEKO-Learner is running on FastAPI."}

@app.post("/learn")
async def learn(request: LearnRequest):
    vectorstore = lifespan_context.get("vectorstore")
    if not vectorstore:
        raise HTTPException(status_code=500, detail="Vectorstore is not initialized")
    try:
        vectorstore.add_texts(texts=[request.text_content])
        logging.info(f"新しい知識の学習に成功しました。")
        return {"message": "Learning successful"}
    except Exception as e:
        logging.error(f"/learn エンドポイントでエラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(request: QueryRequest):
    vectorstore = lifespan_context.get("vectorstore")
    if not vectorstore:
        raise HTTPException(status_code=500, detail="Vectorstore is not initialized")
    try:
        docs = vectorstore.similarity_search(request.query_text, k=5)
        retrieved_docs = [doc.page_content for doc in docs]
        logging.info(f"問い合わせ「{request.query_text}」に対して{len(retrieved_docs)}件の情報を返しました。")
        return {"documents": retrieved_docs}
    except Exception as e:
        logging.error(f"/query エンドポイントでエラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize_and_learn(request: SummarizeRequest):
    vectorstore = lifespan_context.get("vectorstore")
    genai_model = lifespan_context.get("genai_model")
    text_splitter = lifespan_context.get("text_splitter")
    if not all([vectorstore, genai_model, text_splitter]):
        raise HTTPException(status_code=500, detail="Application not fully initialized")
    try:
        prompt = f"以下の会話履歴を、未来の自分が文脈を思い出すための、簡潔な箇条書きのメモに要約してください。\n\n# 会話履歴\n{request.history_text}"
        summary_response = genai_model.generate_content(prompt)
        summary_text = summary_response.text
        
        vectorstore.add_texts(texts=[summary_text])
        logging.info(f"会話履歴を要約し、学習しました。")
        return {"message": "Summarize and learn successful"}
    except Exception as e:
        logging.error(f"/summarize エンドポイントでエラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memorize-style")
async def memorize_style(request: StyleRequest):
    supabase_client = lifespan_context.get("supabase_client")
    if not supabase_client:
        raise HTTPException(status_code=500, detail="Supabase client is not initialized")
    try:
        style_data = request.style_data
        insert_data = {
            "style_name": style_data.get('style_analysis', {}).get('style_name', '無題'),
            "source_prompt": style_data.get('source_prompt'),
            "source_image_url": style_data.get('source_image_url'),
            "style_analysis_json": style_data.get('style_analysis')
        }
        supabase_client.table('styles').insert(insert_data).execute()
        logging.info(f"新しいスタイル「{insert_data['style_name']}」を記憶しました。")
        return {"message": "Style memorized successfully"}
    except Exception as e:
        logging.error(f"/memorize-style エンドポイントでエラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/retrieve-styles")
async def retrieve_styles():
    supabase_client = lifespan_context.get("supabase_client")
    if not supabase_client:
        raise HTTPException(status_code=500, detail="Supabase client is not initialized")
    try:
        response = supabase_client.table('styles').select('style_name, source_prompt, source_image_url, style_analysis_json').execute()
        reconstructed_styles = [
            {"source_prompt": item.get("source_prompt"), "source_image_url": item.get("source_image_url"), "style_analysis": item.get("style_analysis_json")} 
            for item in response.data
        ]
        logging.info(f"{len(reconstructed_styles)}件の学習済みスタイルを返しました。")
        return {"learned_styles": reconstructed_styles}
    except Exception as e:
        logging.error(f"/retrieve-styles エンドポイントでエラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

