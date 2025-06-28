# MIRAI-HEKO-Learner/learner_main.py (Ver.7.0 - The Persistent Soul)
# Creator & Partner: imazine & Gemini
# Last Updated: 2025-06-29
# - Added endpoints for character state and vocabulary persistence.

import os
import logging
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional, Dict

import google.generativeai as genai
from supabase.client import Client, create_client
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Context for Lifespan ---
lifespan_context = {}

def get_env_variable(var_name, is_critical=True, default=None):
    """Safely get environment variables."""
    value = os.getenv(var_name)
    if not value and is_critical:
        logging.critical(f"Mandatory environment variable '{var_name}' is not set.")
        raise ValueError(f"'{var_name}' is not set.")
    return value if value else default

# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and teardown resources for the FastAPI app."""
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
            query_name="match_documents",
        )
        lifespan_context["text_splitter"] = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        lifespan_context["genai_model"] = genai.GenerativeModel('gemini-2.5-pro-preview-03-25')

        logging.info("All initializations are complete. Learner is healthy.")
    except Exception as e:
        logging.critical(f"A fatal error occurred during learner initialization: {e}", exc_info=True)
    
    yield
    
    logging.info("Learner's lifecycle is ending.")
    lifespan_context.clear()

app = FastAPI(lifespan=lifespan)

# --- Pydantic Models ---
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
# ★★★ New Models for Persistence ★★★
class CharacterStateUpdate(BaseModel):
    states: Dict[str, str]
class VocabularyUpdate(BaseModel):
    words_used: List[str]


# --- API Endpoints ---

@app.post("/learn")
async def learn(request: TextContent):
    """Learn from a new text document."""
    if "vectorstore" not in lifespan_context:
        raise HTTPException(status_code=500, detail="Vectorstore is not initialized")
    try:
        texts = lifespan_context["text_splitter"].split_text(request.text_content)
        lifespan_context["vectorstore"].add_texts(texts=texts)
        logging.info(f"Learned {len(texts)} new chunks.")
        return {"message": "Learning successful"}
    except Exception as e:
        logging.error(f"Error in /learn: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(request: QueryRequest):
    """Query the vector store for relevant documents."""
    if "vectorstore" not in lifespan_context:
        raise HTTPException(status_code=500, detail="Vectorstore not initialized")
    try:
        docs = lifespan_context["vectorstore"].similarity_search(
            request.query_text, 
            k=request.k,
            filter=request.filter or {}
        )
        logging.info(f"Returned {len(docs)} documents for query: '{request.query_text}'")
        return {"documents": [doc.page_content for doc in docs]}
    except Exception as e:
        logging.error(f"Error in /query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    """Summarize conversation history and add it to memory."""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"以下の会話履歴を、次の会話で参照しやすいように、重要なキーワードや出来事を箇条書きで簡潔に要約してください。\n\n# 会話履歴\n{request.history_text}"
        response = await model.generate_content_async(prompt)
        summary_text = response.text.strip()

        lifespan_context["vectorstore"].add_texts(texts=[f"最近の会話の要約: {summary_text}"])
        logging.info("Saved conversation summary to vector DB.")

        return {"summary": summary_text}
    except Exception as e:
        logging.error(f"Error in /summarize: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ★★★ New Endpoint: Get Character States ★★★
@app.get("/character-states")
async def get_character_states():
    """Retrieve the current states of the characters from Supabase."""
    try:
        response = lifespan_context["supabase_client"].table('character_states').select('*').limit(1).single().execute()
        logging.info("Fetched character states from DB.")
        return response.data
    except Exception as e:
        logging.error(f"Error getting character states: {e}", exc_info=True)
        # Return a default state if not found or on error, to ensure bot can always start
        return {"last_interaction_summary": "まだ会話が始まっていません。", "mirai_mood": "ニュートラル", "heko_mood": "ニュートラル"}

# ★★★ New Endpoint: Update Character States ★★★
@app.post("/character-states")
async def update_character_states(request: CharacterStateUpdate):
    """Update or insert character states in Supabase."""
    try:
        # Upsert operation: update if exists, insert if not. `id`=1 is a fixed row.
        lifespan_context["supabase_client"].table('character_states').upsert({'id': 1, **request.states}).execute()
        logging.info(f"Updated character states in DB: {request.states}")
        return {"message": "State updated successfully"}
    except Exception as e:
        logging.error(f"Error updating character states: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ★★★ New Endpoint: Get Vocabulary ★★★
@app.get("/vocabulary")
async def get_vocabulary():
    """Retrieve the vocabulary from Supabase."""
    try:
        response = lifespan_context["supabase_client"].table('gals_words').select('*').order('total', desc=True).execute()
        return {"vocabulary": response.data}
    except Exception as e:
        logging.error(f"Error getting vocabulary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ★★★ New Endpoint: Update Vocabulary ★★★
@app.post("/vocabulary/update")
async def update_vocabulary(request: VocabularyUpdate):
    """Increment usage counts for words used in a conversation."""
    supabase: Client = lifespan_context["supabase_client"]
    try:
        for word in request.words_used:
            # This is a simplified logic. For production, a more robust RPC function in Supabase would be better.
            # 1. Fetch current counts
            current_data = supabase.table('gals_words').select('total').eq('word', word).single().execute()
            if current_data.data:
                # 2. Increment and update
                new_total = current_data.data['total'] + 1
                supabase.table('gals_words').update({'total': new_total}).eq('word', word).execute()
        logging.info(f"Updated vocabulary for words: {request.words_used}")
        return {"message": "Vocabulary updated"}
    except Exception as e:
        logging.error(f"Error updating vocabulary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# --- Other Endpoints (Unchanged) ---
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
        logging.info(f"Logged learning history: {request.filename} by {request.username}")
        return {"message": "Learning history logged successfully"}
    except Exception as e:
        logging.error(f"Error in /log-learning-history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

