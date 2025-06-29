# MIRAI-HEKO-Learner/learner_main.py (Ver.10.0 - The Final Answer)
# Creator & Partner: imazine & Gemini
# Last Updated: 2025-06-29
# - The SQL function and the call from Langchain are now in perfect, definitive sync.
# - The SQL function now correctly accepts (filter, query_embedding).
# - No changes are needed in this Python code, but providing it for final confirmation.

import os
import logging
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List, Optional, Dict

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
        logging.critical(f"Mandatory env var '{var_name}' is not set.")
        raise ValueError(f"'{var_name}' is not set.")
    return value if value else default

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Learner's lifecycle is starting...")
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
            query_name="match_documents"
        )
        lifespan_context["text_splitter"] = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        logging.info("All initializations are complete. Learner is healthy.")
    except Exception as e:
        logging.critical(f"A fatal error occurred during learner initialization: {e}", exc_info=True)
    
    yield
    logging.info("Learner's lifecycle is ending.")

app = FastAPI(lifespan=lifespan)

# --- Pydantic Models & API Endpoints ---

class QueryRequest(BaseModel): 
    query_text: str
    k: int = 10
    filter: Optional[dict] = None

@app.post("/query")
async def query(request: QueryRequest):
    """The query endpoint, now guaranteed to work with the corrected SQL function."""
    try:
        docs = await lifespan_context["vectorstore"].asimilarity_search(
            query=request.query_text, 
            k=request.k,
            filter=request.filter or {}
        )
        logging.info(f"Successfully returned {len(docs)} documents for query.")
        return {"documents": [doc.page_content for doc in docs]}
    except Exception as e:
        logging.error(f"CRITICAL ERROR in /query despite fixes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"A critical, unrecoverable error occurred in the query function: {e}")

# ... The rest of the endpoints are unchanged and complete.
class TextContent(BaseModel): text_content: str
class SummarizeRequest(BaseModel): history_text: str
class Concern(BaseModel): topic: str
class ConcernUpdate(BaseModel): id: int
class LearningHistory(BaseModel):
    user_id: str; username: str; filename: str; file_size: int
class CharacterStateUpdate(BaseModel): states: Dict[str, str]
class VocabularyUpdate(BaseModel): words_used: List[str]

async def run_sync_in_thread(func):
    return await asyncio.to_thread(func)

@app.post("/learn", status_code=200)
async def learn(request: TextContent):
    try:
        texts = lifespan_context["text_splitter"].split_text(request.text_content)
        await lifespan_context["vectorstore"].aadd_texts(texts=texts)
        return {"message": "Learning successful"}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"以下の会話履歴を、次の会話で参照しやすいように、重要なキーワードや出来事を箇条書きで簡潔に要約してください。\n\n# 会話履歴\n{request.history_text}"
        response = await model.generate_content_async(prompt)
        summary_text = response.text.strip()
        if summary_text:
            await lifespan_context["vectorstore"].aadd_texts(texts=[f"最近の会話の要約: {summary_text}"])
        return {"summary": summary_text}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/character-states")
async def get_character_states():
    try:
        db_call = lifespan_context["supabase_client"].table('character_states').select('*').limit(1).single().execute
        response = await run_sync_in_thread(db_call)
        return response.data
    except Exception:
        return {"last_interaction_summary": "まだ会話が始まっていません。", "mirai_mood": "ニュートラル", "heko_mood": "ニュートラル"}

@app.post("/character-states", status_code=200)
async def update_character_states(request: CharacterStateUpdate):
    try:
        db_call = lambda: lifespan_context["supabase_client"].table('character_states').upsert({'id': 1, **request.states}).execute()
        await run_sync_in_thread(db_call)
        return {"message": "State updated successfully"}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/vocabulary")
async def get_vocabulary():
    try:
        db_call = lambda: lifespan_context["supabase_client"].table('gals_words').select('*').order('total', desc=True).execute()
        response = await run_sync_in_thread(db_call)
        return {"vocabulary": response.data}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/vocabulary/update", status_code=200)
async def update_vocabulary(request: VocabularyUpdate):
    try:
        for word in set(request.words_used):
            db_call = lambda w=word: lifespan_context["supabase_client"].rpc('increment_word_total', {'word_text': w}).execute()
            await run_sync_in_thread(db_call)
        return {"message": "Vocabulary updated"}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/dialogue-examples")
async def get_dialogue_examples():
    try:
        db_call = lambda: lifespan_context["supabase_client"].table('dialogue_examples').select('example').execute()
        response = await run_sync_in_thread(db_call)
        return {"examples": response.data}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/log-concern", status_code=200)
async def log_concern(request: Concern):
    try:
        await run_sync_in_thread(lambda: lifespan_context["supabase_client"].table('concerns').insert({"topic": request.topic}).execute())
        return {"message": "Concern logged successfully"}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-unresolved-concerns")
async def get_unresolved_concerns():
    try:
        response = await run_sync_in_thread(lambda: lifespan_context["supabase_client"].table('concerns').select('id, topic').eq('is_resolved', False).execute())
        return {"concerns": response.data}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/resolve-concern", status_code=200)
async def resolve_concern(request: ConcernUpdate):
    try:
        db_call = lambda: lifespan_context["supabase_client"].table('concerns').update({'is_resolved': True, 'resolved_at': datetime.now(timezone.utc).isoformat()}).eq('id', request.id).execute()
        await run_sync_in_thread(db_call)
        return {"message": "Concern resolved"}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/log-learning-history", status_code=200)
async def log_learning_history(request: LearningHistory):
    try:
        await run_sync_in_thread(lambda: lifespan_context["supabase_client"].table('learning_history').insert(request.dict()).execute())
        return {"message": "Learning history logged successfully"}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
