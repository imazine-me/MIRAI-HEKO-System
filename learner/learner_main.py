# MIRAI-HEKO-Learner/learner_main.py (Ver.5.0 - The Sentient Soul)

import os
import logging
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional

import google.generativeai as genai
from supabase.client import Client, create_client
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

lifespan_context = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("学習係のライフサイクルが開始します...")
    try:
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')

        if not all([gemini_api_key, supabase_url, supabase_key]):
            raise ValueError("必須の環境変数が設定されていません。")

        lifespan_context["supabase_client"] = create_client(supabase_url, supabase_key)
        logging.info("Supabaseクライアントの初期化に成功。")

        genai.configure(api_key=gemini_api_key)
        lifespan_context["genai_model"] = genai.GenerativeModel('gemini-2.5-pro-preview-03-25')
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-004", google_api_key=gemini_api_key)
        
        lifespan_context["vectorstore"] = SupabaseVectorStore(
            client=lifespan_context["supabase_client"],
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents"
        )
        logging.info("全ての初期化処理が完了。学習係は正常です。")
    except Exception as e:
        logging.critical(f"初期設定中に致命的なエラー: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize: {e}")
    yield
    logging.info("学習係のライフサイクルが終了します。")

app = FastAPI(lifespan=lifespan)

# --- Pydanticモデル定義 ---
class TextContent(BaseModel): text_content: str
class Query(BaseModel): query_text: str
class SummarizeRequest(BaseModel): history_text: str
class Concern(BaseModel): topic: str
class ConcernUpdate(BaseModel): id: int
class GrowthReportRequest(BaseModel): summaries: List[str]

# --- APIエンドポイント ---
@app.get("/")
async def index(): return {"message": "MIRAI-HEKO-Learner is running."}

@app.post("/learn")
async def learn(request: TextContent):
    try:
        lifespan_context["vectorstore"].add_texts(texts=[request.text_content])
        return {"message": "Learning successful"}
    except Exception as e:
        logging.error(f"Error in /learn: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(request: Query):
    try:
        docs = lifespan_context["vectorstore"].similarity_search(request.query_text, k=10)
        return {"documents": [doc.page_content for doc in docs]}
    except Exception as e:
        logging.error(f"Error in /query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    try:
        prompt = f"以下の会話履歴を、未来の自分が文脈を思い出すための、簡潔な箇条書きのメモに要約してください。\n\n# 会話履歴\n{request.history_text}"
        summary = lifespan_context["genai_model"].generate_content(prompt).text
        lifespan_context["vectorstore"].add_texts(texts=[summary])
        return {"message": "Summarize and learn successful", "summary": summary}
    except Exception as e:
        logging.error(f"Error in /summarize: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/log-concern")
async def log_concern(request: Concern):
    try:
        lifespan_context["supabase_client"].table('concerns').insert({"topic": request.topic}).execute()
        return {"message": "Concern logged successfully"}
    except Exception as e:
        logging.error(f"Error in /log-concern: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-unresolved-concerns")
async def get_unresolved_concerns():
    try:
        response = lifespan_context["supabase_client"].table('concerns').select('*').filter('is_resolved', 'eq', 'false').execute()
        return {"concerns": response.data}
    except Exception as e:
        logging.error(f"Error in /get-unresolved-concerns: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/resolve-concern")
async def resolve_concern(request: ConcernUpdate):
    try:
        lifespan_context["supabase_client"].table('concerns').update({'is_resolved': True, 'notified_at': datetime.now(timezone.utc).isoformat()}).eq('id', request.id).execute()
        return {"message": "Concern resolved"}
    except Exception as e:
        logging.error(f"Error in /resolve-concern: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
