# MIRAI-HEKO-Learner/learner_main.py (Ver.Ω - The Omega)
# Creator & Partner: imazine & Gemini
# Finalized with the ultimate insights from a fellow AI.
# This version is stable, robust, and truly production-ready.

import os
import logging
import asyncio
import hashlib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

import google.generativeai as genai
from supabase.client import Client, create_client
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Context & Optimized Thread Pool ---
lifespan_context: Dict[str, Any] = {}
# ★★★ To prevent resource exhaustion, the thread pool size is explicitly controlled. ★★★
# This formula provides a safe number of workers based on available CPU cores.
thread_executor = ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 1) * 2), thread_name_prefix="learner_sync_worker")

def get_env_variable(var_name: str, is_critical: bool = True, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(var_name)
    if not value and is_critical:
        logging.critical(f"Mandatory env var '{var_name}' is not set.")
        raise ValueError(f"'{var_name}' is not set.")
    return value if value else default

# --- Helper for Truly Async DB calls ---
async def run_sync_in_thread(func, *args, **kwargs) -> Any:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(thread_executor, lambda: func(*args, **kwargs))

# --- Startup Check ---
async def check_rpc_signature(db_client: Client):
    """Asserts that the 'match_documents' function is callable via a dummy RPC call."""
    try:
        logging.info("Performing a startup check on the 'match_documents' DB function...")
        dummy_embedding = [0.0] * 768
        # ★★★ CRITICAL FIX: The .execute() call must be inside the lambda to be run in the thread. ★★★
        await run_sync_in_thread(
            lambda: db_client.rpc('match_documents', {'query_embedding': dummy_embedding}).execute()
        )
        logging.info("Database function 'match_documents' signature check passed successfully.")
    except Exception as e:
        logging.critical(f"DATABASE SIGNATURE CHECK FAILED: {e}. The 'match_documents' function in Supabase is likely incorrect. Please apply the latest SQL patch.", exc_info=True)
        raise

# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Learner's lifecycle is starting...")
    try:
        supabase_url = get_env_variable("SUPABASE_URL")
        supabase_key = get_env_variable("SUPABASE_KEY")
        gemini_api_key = get_env_variable("GEMINI_API_KEY")

        supabase_client = create_client(supabase_url, supabase_key)
        lifespan_context["supabase_client"] = supabase_client
        genai.configure(api_key=gemini_api_key)
        
        await check_rpc_signature(supabase_client)

        lifespan_context["embeddings"] = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
        lifespan_context["vectorstore"] = SupabaseVectorStore(client=supabase_client, embedding=lifespan_context["embeddings"], table_name="documents", query_name="match_documents")
        # ★★★ Cleanup: length_function=len is default and can be removed for clarity. ★★★
        lifespan_context["text_splitter"] = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        
        logging.info("All initializations are complete. Learner is healthy.")
    except Exception as e:
        # This will now catch the startup check failure and prevent a broken launch.
        logging.critical(f"A fatal error occurred during learner initialization: {e}", exc_info=True)
        # In a real K8s/Docker environment, this would cause the container to fail and restart.
        raise SystemExit(1)
    
    yield
    
    thread_executor.shutdown(wait=True)
    logging.info("Learner's lifecycle is ending and thread pool is shut down.")

app = FastAPI(lifespan=lifespan)

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query_text: str
    k: int = 10
    filter: Dict = Field(default_factory=dict)

class SimilarityResponse(BaseModel):
    content: str
    metadata: Optional[Dict] = None
    similarity: float

class TextContent(BaseModel):
    text_content: str
    
# ★★★ CRITICAL FIX: The missing model that caused NameError ★★★
class SummarizeRequest(BaseModel):
    history_text: str

# ... (Other models are unchanged)

# --- API Endpoints ---
@app.post("/query", response_model=List[SimilarityResponse])
async def query(request: QueryRequest):
    """The query endpoint, now using the correct synchronous method in a thread and returning similarity."""
    try:
        vectorstore = lifespan_context['vectorstore']
        # Use getattr for fallback between langchain versions for better compatibility
        search_fn = getattr(vectorstore, 'similarity_search_with_relevance_scores', vectorstore.similarity_search_with_score)
        
        results = await run_sync_in_thread(search_fn, request.query_text, k=request.k, filter=request.filter)
        
        logging.info(f"Successfully returned {len(results)} documents for query.")
        return [
            SimilarityResponse(content=doc.page_content, metadata=doc.metadata, similarity=score)
            for doc, score in results
        ]
    except Exception as e:
        logging.error(f"CRITICAL ERROR in /query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/learn", status_code=200)
async def learn(request: TextContent):
    try:
        # Guard against Out-of-Memory on very large inputs
        texts = lifespan_context["text_splitter"].split_text(request.text_content)[:500]
        if not texts:
            return {"message": "No text to learn."}

        # Add unique IDs to prevent duplicate chunk registration
        ids = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        await run_sync_in_thread(lifespan_context["vectorstore"].add_texts, texts=texts, ids=ids)
        logging.info(f"Learned and indexed {len(texts)} new chunks.")
        return {"message": "Learning successful"}
    except Exception as e:
        logging.error(f"Error in /learn: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"以下の会話履歴を、次の会話で参照しやすいように、重要なキーワードや出来事を箇条書きで簡潔に要約してください。\n\n# 会話履歴\n{request.history_text}"
        response = await model.generate_content_async(prompt)
        summary_text = response.text.strip()
        if summary_text:
            # ★★★ FIX: .add_texts is also sync ★★★
            await run_sync_in_thread(
                lifespan_context["vectorstore"].add_texts,
                texts=[f"最近の会話の要約: {summary_text}"]
            )
            logging.info("Saved conversation summary to vector DB.")
        return {"summary": summary_text}
    except Exception as e:
        logging.error(f"Error in /summarize: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ... (The rest of the endpoints are also wrapped for safety)
@app.get("/character-states")
async def get_character_states():
    try:
        db_call = lambda: lifespan_context["supabase_client"].table('character_states').select('*').limit(1).single().execute()
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
