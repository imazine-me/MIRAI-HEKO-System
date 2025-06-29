# MIRAI-HEKO-Learner/learner_main.py (Ver.Ω-Kai - The Restored Soul)
# Creator &amp; Partner: imazine &amp; Gemini
# Last Updated: 2025-06-29
# - Added all endpoints required to restore the beloved features.

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
from supabase.client import Client, create\_client
from langchain\_google\_genai import GoogleGenerativeAIEmbeddings
from langchain\_community.vectorstores import SupabaseVectorStore
from langchain.text\_splitter import CharacterTextSplitter
from dotenv import load\_dotenv

load\_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# \--- Global Context & Optimized Thread Pool ---

lifespan\_context: Dict[str, Any] = {}
thread\_executor = ThreadPoolExecutor(max\_workers=min(8, (os.cpu\_count() or 1) \* 2), thread\_name\_prefix="learner\_sync\_worker")

def get\_env\_variable(var\_name: str, is\_critical: bool = True, default: Optional[str] = None) -\> Optional[str]:
value = os.getenv(var\_name)
if not value and is\_critical:
logging.critical(f"Mandatory env var '{var\_name}' is not set.")
raise ValueError(f"'{var\_name}' is not set.")
return value if value else default

# \--- Helper for Truly Async DB calls ---

async def run\_sync\_in\_thread(func, \*args, \*\*kwargs) -\> Any:
loop = asyncio.get\_running\_loop()
return await loop.run\_in\_executor(thread\_executor, lambda: func(\*args, \*\*kwargs))

# \--- Startup Check ---

async def check\_rpc\_signature(db\_client: Client):
"""Asserts that the 'match\_documents' function is callable via a dummy RPC call."""
try:
logging.info("Performing a startup check on the 'match\_documents' DB function...")
dummy\_embedding = [0.0] \* 768
await run\_sync\_in\_thread(
lambda: db\_client.rpc('match\_documents', {'query\_embedding': dummy\_embedding}).execute()
)
logging.info("Database function 'match\_documents' signature check passed successfully.")
except Exception as e:
logging.critical(f"DATABASE SIGNATURE CHECK FAILED: {e}. The 'match\_documents' function in Supabase is likely incorrect. Please apply the latest SQL patch.", exc\_info=True)
raise

# \--- Lifespan Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
logging.info("Learner's lifecycle is starting...")
try:
supabase\_url = get\_env\_variable("SUPABASE\_URL")
supabase\_key = get\_env\_variable("SUPABASE\_KEY")
gemini\_api\_key = get\_env\_variable("GEMINI\_API\_KEY")

supabase_client = create_client(supabase_url, supabase_key)
lifespan_context["supabase_client"] = supabase_client
genai.configure(api_key=gemini_api_key)

await check_rpc_signature(supabase_client)

lifespan_context["embeddings"] = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
lifespan_context["vectorstore"] = SupabaseVectorStore(client=supabase_client, embedding=lifespan_context["embeddings"], table_name="documents", query_name="match_documents")
lifespan_context["text_splitter"] = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)

logging.info("All initializations are complete. Learner is healthy.")
except Exception as e:
logging.critical(f"A fatal error occurred during learner initialization: {e}", exc_info=True)
raise SystemExit(1)

yield

thread_executor.shutdown(wait=True)
logging.info("Learner's lifecycle is ending and thread pool is shut down.")


app = FastAPI(lifespan=lifespan)

# \--- Pydantic Models ---

class QueryRequest(BaseModel):
query\_text: str
k: int = 10
filter: Dict = Field(default\_factory=dict)

class SimilarityResponse(BaseModel):
content: str
metadata: Optional[Dict] = None
similarity: float

class TextContent(BaseModel):
text\_content: str

class SummarizeRequest(BaseModel):
history\_text: str

class Concern(BaseModel):
user\_id: str
topic: str

class ConcernUpdate(BaseModel):
id: int

class LearningHistory(BaseModel):
user\_id: str
username: str
filename: str
file\_size: int

class StyleData(BaseModel):
user\_id: str
style\_name: str
style\_keywords: List[str]
style\_description: str
source\_prompt: str
source\_image\_url: str

# \--- API Endpoints ---

@app.post("/query", response\_model=List[SimilarityResponse])
async def query(request: QueryRequest):
try:
vectorstore = lifespan\_context['vectorstore']
search\_fn = getattr(vectorstore, 'similarity\_search\_with\_relevance\_scores', vectorstore.similarity\_search\_with\_score)
results = await run\_sync\_in\_thread(search\_fn, request.query\_text, k=request.k, filter=request.filter)
logging.info(f"Successfully returned {len(results)} documents for query.")
return [
SimilarityResponse(content=doc.page\_content, metadata=doc.metadata, similarity=score)
for doc, score in results
]
except Exception as e:
logging.error(f"CRITICAL ERROR in /query: {e}", exc\_info=True)
raise HTTPException(status\_code=500, detail=str(e))

@app.post("/learn", status\_code=200)
async def learn(request: TextContent):
try:
texts = lifespan\_context["text\_splitter"].split\_text(request.text\_content)[:500]
if not texts: return {"message": "No text to learn."}
ids = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
await run\_sync\_in\_thread(lifespan\_context["vectorstore"].add\_texts, texts=texts, ids=ids)
logging.info(f"Learned and indexed {len(texts)} new chunks.")
return {"message": "Learning successful"}
except Exception as e:
logging.error(f"Error in /learn: {e}", exc\_info=True)
raise HTTPException(status\_code=500, detail=str(e))

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
try:
model = genai.GenerativeModel('gemini-2.0-flash')
prompt = f"以下の会話履歴を、次の会話で参照しやすいように、重要なキーワードや出来事を箇条書きで簡潔に要約してください。\\n\\n\# 会話履歴\\n{request.history\_text}"
response = await model.generate\_content\_async(prompt)
summary\_text = response.text.strip()
if summary\_text:
await run\_sync\_in\_thread(
lifespan\_context["vectorstore"].add\_texts,
texts=[f"最近の会話の要約: {summary\_text}"]
)
return {"summary": summary\_text}
except Exception as e: raise HTTPException(status\_code=500, detail=str(e))

# ★★★ ここからが、失われた記憶を取り戻すための、新しい窓口です ★★★

@app.post("/log-learning-history", status\_code=200)
async def log\_learning\_history(request: LearningHistory):
"""「\!learn」コマンドの履歴を記録します"""
try:
await run\_sync\_in\_thread(
lambda: lifespan\_context["supabase\_client"].table('learning\_history').insert(request.dict()).execute()
)
logging.info(f"Logged learning history for {request.filename}")
return {"message": "Learning history logged successfully"}
except Exception as e:
logging.error(f"Error in /log-learning-history: {e}", exc\_info=True)
raise HTTPException(status\_code=500, detail=str(e))

@app.post("/memorize-style", status\_code=200)
async def memorize\_style(request: StyleData):
"""「🎨」リアクションで学習した画風をデータベースに保存します"""
try:
await run\_sync\_in\_thread(
lambda: lifespan\_context["supabase\_client"].table('learned\_styles').insert(request.dict()).execute()
)
logging.info(f"Memorized new style: {request.style\_name}")
return {"message": "Style memorized successfully"}
except Exception as e:
logging.error(f"Error in /memorize-style: {e}", exc\_info=True)
raise HTTPException(status\_code=500, detail=str(e))

@app.get("/retrieve-styles", response\_model=List[Dict])
async def retrieve\_styles(user\_id: str):
"""特定のユーザーが学習させた画風のリストを取得します"""
try:
response = await run\_sync\_in\_thread(
lambda: lifespan\_context["supabase\_client"].table('learned\_styles').select('\*').eq('user\_id', user\_id).execute()
)
return response.data
except Exception as e:
logging.error(f"Error in /retrieve-styles: {e}", exc\_info=True)
raise HTTPException(status\_code=500, detail=str(e))

@app.post("/log-concern", status\_code=200)
async def log\_concern(request: Concern):
"""へー子の気づかい機能のため、ユーザーの悩みを記録します"""
try:
await run\_sync\_in\_thread(
lambda: lifespan\_context["supabase\_client"].table('concerns').insert(request.dict()).execute()
)
logging.info(f"Logged new concern: {request.topic}")
return {"message": "Concern logged successfully"}
except Exception as e:
logging.error(f"Error in /log-concern: {e}", exc\_info=True)
raise HTTPException(status\_code=500, detail=str(e))

@app.get("/get-unresolved-concerns", response\_model=List[Dict])
async def get\_unresolved\_concerns(user\_id: str):
"""指定されたユーザーの、まだ解決していない悩みのリストを取得します"""
try:
response = await run\_sync\_in\_thread(
lambda: lifespan\_context["supabase\_client"].table('concerns').select('\*').eq('user\_id', user\_id).eq('is\_resolved', False).execute()
)
return response.data
except Exception as e:
logging.error(f"Error in /get-unresolved-concerns: {e}", exc\_info=True)
raise HTTPException(status\_code=500, detail=str(e))

@app.post("/resolve-concern", status\_code=200)
async def resolve\_concern(request: ConcernUpdate):
"""悩みが解決したことをマークします"""
try:
await run\_sync\_in\_thread(
lambda: lifespan\_context["supabase\_client"].table('concerns').update({'is\_resolved': True, 'resolved\_at': datetime.now(timezone.utc).isoformat()}).eq('id', request.id).execute()
)
logging.info(f"Resolved concern with id: {request.id}")
return {"message": "Concern resolved"}
except Exception as e:
logging.error(f"Error in /resolve-concern: {e}", exc\_info=True)
raise HTTPException(status\_code=500, detail=str(e))
