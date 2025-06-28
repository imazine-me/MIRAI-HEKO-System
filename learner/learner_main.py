# MIRAI-HEKO-Learner/learner_main.py (Ver.6.2 - The Final Truth)
import os
import logging
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional

import google.generativeai as genai
from supabase.client import Client, create_client
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

lifespan_context = {}

def get_env_variable(var_name, is_critical=True, default=None):
    value = os.getenv(var_name)
    if not value and is_critical:
        logging.critical(f"必須の環境変数 '{var_name}' が設定されていません。")
        raise ValueError(f"'{var_name}' is not set.")
    return value if value else default

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("学習係のライフサイクルが開始します...")
    try:
        supabase_url = get_env_variable("SUPABASE_URL")
        supabase_key = get_env_variable("SUPABASE_KEY")
        gemini_api_key = get_env_variable("GEMINI_API_KEY")

        lifespan_context["supabase_client"] = create_client(supabase_url, supabase_key)
        genai.configure(api_key=gemini_api_key)
        
        lifespan_context["embeddings"] = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
        
        lifespan_context["vectorstore"] = SupabaseVectorStore(
            client=lifespan_context["supabase_client"],
            embedding=lifespan_context["embeddings"],
            table_name="documents",
            query_name="match_documents",
        )
        lifespan_context["text_splitter"] = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        lifespan_context["genai_model"] = genai.GenerativeModel('gemini-2.5-pro-preview-03-25')

        logging.info("全ての初期化処理が完了。学習係は正常です。")

    except Exception as e:
        logging.critical(f"学習係の初期化中に致命的なエラーが発生しました: {e}", exc_info=True)
    
    yield
    
    logging.info("学習係のライフサイクルが終了します。")
    lifespan_context.clear()

app = FastAPI(lifespan=lifespan)

class TextContent(BaseModel): text_content: str
class QueryRequest(BaseModel): 
    query_text: str
    k: int = 10
    filter: Optional[dict] = None
class SummarizeRequest(BaseModel): history_text: str
class Concern(BaseModel): topic: str
class ConcernUpdate(BaseModel): id: int
class LearningHistory(BaseModel):
    user_id: str
    username: str
    filename: str
    file_size: int

@app.post("/learn")
async def learn(request: TextContent):
    if "vectorstore" not in lifespan_context:
        raise HTTPException(status_code=500, detail="Vectorstore is not initialized")
    try:
        texts = lifespan_context["text_splitter"].split_text(request.text_content)
        lifespan_context["vectorstore"].add_texts(texts=texts)
        logging.info(f"{len(texts)}個のチャンクを学習しました。")
        # (学習内容のタイトル生成と保存ロジックもここに含める)
        return {"message": "Learning successful"}
    except Exception as e:
        logging.error(f"Error in /learn: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(request: QueryRequest):
    if "vectorstore" not in lifespan_context:
        raise HTTPException(status_code=500, detail="Vectorstore not initialized")
    try:
        docs = lifespan_context["vectorstore"].similarity_search(
            request.query_text, 
            k=request.k,
            filter=request.filter or {} # filterがNoneの場合、空の辞書を渡す
        )
        logging.info(f"問い合わせ「{request.query_text}」に対して{len(docs)}件の情報を返しました。")
        return {"documents": [doc.page_content for doc in docs]}
    except Exception as e:
        logging.error(f"Error in /query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"以下の会話履歴を、次の会話で参照しやすいように、重要なキーワードや出来事を箇条書きで簡潔に要約してください。\n\n# 会話履歴\n{request.history_text}"
        response = await model.generate_content_async(prompt)
        summary_text = response.text.strip()

        # 要約結果もベクトルDBに保存
        lifespan_context["vectorstore"].add_texts(texts=[f"最近の会話の要約: {summary_text}"])
        logging.info("会話の要約をベクトルDBに保存しました。")

        return {"summary": summary_text}
    except Exception as e:
        logging.error(f"Error in /summarize: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- 残りのエンドポイント (log-concern, get-unresolved-concerns, resolve-concern, log-learning-history) ---
# これらの関数は変更ありません
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
        response = lifespan_context["supabase_client"].table('concerns').select('*').eq('is_resolved', False).execute()
        return {"concerns": response.data}
    except Exception as e:
        logging.error(f"Error in /get-unresolved-concerns: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/resolve-concern")
async def resolve_concern(request: ConcernUpdate):
    try:
        lifespan_context["supabase_client"].table('concerns').update({'is_resolved': True, 'resolved_at': datetime.now(timezone.utc).isoformat()}).eq('id', request.id).execute()
        return {"message": "Concern resolved"}
    except Exception as e:
        logging.error(f"Error in /resolve-concern: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/log-learning-history")
async def log_learning_history(request: LearningHistory):
    try:
        lifespan_context["supabase_client"].table('learning_history').insert(request.dict()).execute()
        logging.info(f"学習履歴を記録しました: {request.filename} by {request.username}")
        return {"message": "Learning history logged successfully"}
    except Exception as e:
        logging.error(f"Error in /log-learning-history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
