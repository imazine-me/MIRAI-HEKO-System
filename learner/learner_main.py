# learner_main.py (ver.Ω+ - The True Final Version)
# The Temple of Soul and Memory, running on Supabase.
# Part 1/2: Core Setup and Memory I/O

import os
import logging
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import datetime as dt

# LangChain and Vector Store related imports
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Supabase Client
from supabase.client import Client, create_client

# --- 1. 初期設定 (Initial Setup) ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
app = FastAPI(
    title="Learner API - The Soul of MIRAI-HEKO-Bot",
    description="This API manages the long-term memory, style palette, character states, and concerns.",
    version="2.0.0"
)

# --- 2. クライアントの初期化 (Client Initialization) ---

try:
    # Supabaseクライアントを環境変数から初期化
    supabase_url: str = os.environ.get("SUPABASE_URL")
    supabase_key: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") # セキュリティのためサービスキーを使用
    if not supabase_url or not supabase_key:
        raise ValueError("SupabaseのURLまたはサービスロールキーが設定されていません。")
    supabase: Client = create_client(supabase_url, supabase_key)
    logging.info("Supabase client initialized successfully.")

    # GoogleのEmbeddingモデルを初期化 (知識のベクトル化に使用)
    # 動作させる環境のAPIキー設定に依存
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEYが設定されていません。")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document", google_api_key=google_api_key)
    logging.info("Google Generative AI Embeddings initialized successfully.")

except Exception as e:
    logging.critical(f"FATAL: クライアントの初期化中にエラーが発生しました: {e}")
    raise e

# --- 3. テキスト分割とベクトルストアの設定 (Text Splitter and Vector Store Configuration) ---

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True,
)

# ChromaDBをベクトルストア（記憶の大図書館）として利用
# Supabase Functionsのファイルシステムは一時的なものなので、永続化には注意が必要。
# Supabase Storageや外部の永続ディスクサービスと連携するのが理想的。
vectorstore = Chroma(
    collection_name="memory_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db_persistent"
)
logging.info(f"Vector store initialized. Collection: {vectorstore._collection.name}")


# --- 4. APIリクエスト/レスポンスモデル定義 (Pydantic Models) ---

class LearnRequest(BaseModel):
    text_content: str = Field(..., description="学習させたいテキスト本文。")
    metadata: Dict[str, Any] = Field({}, description="テキストに関連するメタデータ（例: {'source': 'file.txt'})。")

class LearnResponse(BaseModel):
    status: str = "success"
    message: str
    learned_chunks: int

class QueryRequest(BaseModel):
    query_text: str = Field(..., description="記憶を検索するための問い合わせテキスト。")

class QueryResponse(BaseModel):
    status: str = "success"
    documents: List[str] = Field(..., description="検索クエリに最も関連性の高い記憶の断片リスト。")


# --- 5. 基本的な記憶の読み書きAPI (Core Memory Endpoints) ---

@app.post("/learn", response_model=LearnResponse, tags=["Memory"])
async def learn_document(request: LearnRequest):
    """
    新しい知識（テキスト）を学習し、ベクトル化して記憶の神殿に保管する。
    また、学習した記録を`learning_log`テーブルに保存する。
    """
    if not request.text_content.strip():
        raise HTTPException(status_code=400, detail="学習するテキスト内容が空です。")
    try:
        logging.info(f"新しい知識の学習を開始します。ソース: {request.metadata.get('source', 'Unknown')}")
        
        docs = text_splitter.create_documents([request.text_content], metadatas=[request.metadata])
        vectorstore.add_documents(docs)
        
        try:
             vectorstore.persist()
             logging.info("Vector storeの永続化に成功しました。")
        except Exception as persist_error:
             logging.warning(f"Vector storeの永続化中に警告: {persist_error}。Supabase Functionsのファイルシステムは一時的な可能性があります。")

        logging.info(f"{len(docs)}個のチャンクをベクトルストアに正常に追加しました。")

        try:
            response = supabase.table('learning_log').insert({
                "content_snippet": request.text_content[:250],
                "metadata": request.metadata
            }).execute()
            logging.info("Supabaseの`learning_log`への記録に成功しました。")
        except Exception as db_error:
            logging.warning(f"Supabaseの`learning_log`への記録中に警告: {db_error}")

        return LearnResponse(
            message="Knowledge successfully acquired and stored in the vector temple.",
            learned_chunks=len(docs)
        )
    except Exception as e:
        logging.error(f"学習処理(/learn)中にエラーが発生しました: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/query", response_model=QueryResponse, tags=["Memory"])
async def query_memory(request: QueryRequest):
    """
    問い合わせ内容に基づいて、記憶の神殿から最も関連性の高い記憶を検索して返す。
    """
    if not request.query_text.strip():
        raise HTTPException(status_code=400, detail="検索クエリが空です。")
    try:
        logging.info(f"記憶の検索を実行します。クエリ: 「{request.query_text}」")
        
        docs = vectorstore.similarity_search(request.query_text, k=5)
        response_docs = [doc.page_content for doc in docs]
        
        logging.info(f"問い合わせに対して{len(response_docs)}件の関連する記憶を返却します。")
        return QueryResponse(documents=response_docs)
    except Exception as e:
        logging.error(f"記憶の検索(/query)中にエラーが発生しました: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/", tags=["System"])
async def root():
    """
    APIサーバーの生存確認用エンドポイント。
    """
    return {"message": "Learner is awake. The soul of imazine's world is waiting for a command."}

# learner_main.py (ver.Ω+ - The True Final Version)
# Part 2/2: Advanced Functions for Style, Emotion, and Growth

# --- 6. 高度な機能のAPIモデル定義 (Pydantic Models for Advanced Features) ---

class StyleLearnRequest(BaseModel):
    image_url: str = Field(..., description="学習させたい画風の画像URL。")
    source_prompt: Optional[str] = Field("", description="その画像が生成された際の、元のプロンプト（もしあれば）。")

class StyleLearnResponse(BaseModel):
    status: str = "success"
    message: str
    learned_style_id: int

class CharacterState(BaseModel):
    user_id: str = "imazine"
    character_name: str = Field(..., description="対象キャラクター名（'みらい' or 'へー子'）。")
    mood: str = Field(..., description="更新後のキャラクターの機嫌。")
    last_interaction_summary: str = Field(..., description="直前のimazineとのやり取りの短い要約。")

class CharacterStateUpdateRequest(BaseModel):
    states: List[CharacterState]

class CharacterStateUpdateResponse(BaseModel):
    status: str = "success"
    message: str
    updated_count: int

class CharacterStateQueryResponse(BaseModel):
    status: str = "success"
    states: Dict[str, Dict[str, Any]] = Field(..., description="キャラクター名ごとの現在の状態。")

class Concern(BaseModel):
    user_id: str = "imazine"
    concern_text: str

class ConcernLogResponse(BaseModel):
    status: str = "success"
    concern_id: int

class UnresolvedConcernResponse(BaseModel):
    status: str = "success"
    concerns: List[Dict[str, Any]]

class ResolveConcernRequest(BaseModel):
    concern_id: int


# --- 7. 画風・感情・成長を司るAPI (Endpoints for Style, Emotion, and Growth) ---

@app.post("/learn_style", response_model=StyleLearnResponse, tags=["Style Palette"])
async def learn_style(request: StyleLearnRequest):
    """
    imazineが🎨リアクションを付けた画像のスタイルを学習し、`style_palette`に保存する。
    """
    try:
        logging.info(f"新しい画風の学習を開始します。ソースURL: {request.image_url}")
        
        # 将来的にはVision APIで画像自体を分析し、キーワードを自動生成する拡張も可能
        style_description = f"Style from image: {request.image_url}"
        if request.source_prompt:
            style_description += f" | Original Prompt: {request.source_prompt}"

        response = supabase.table('style_palette').insert({
            "style_description": style_description,
            "source_url": request.image_url,
            "learned_by": "imazine_reaction"
        }).execute()

        learned_id = response.data[0]['id']
        logging.info(f"新しい画風をID:{learned_id}として`style_palette`に記録しました。")

        return StyleLearnResponse(
            message="Style palette successfully updated with a new artistic soul.",
            learned_style_id=learned_id
        )
    except Exception as e:
        logging.error(f"スタイル学習(/learn_style)中にエラーが発生しました: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/get_style_palette", response_model=QueryResponse, tags=["Style Palette"])
async def get_style_palette():
    """
    現在学習済みの画風（スタイル）の記述リストを取得する。
    """
    try:
        response = supabase.table('style_palette').select("style_description").order('created_at', desc=True).limit(5).execute()

        if not response.data:
            return QueryResponse(documents=[])

        style_descriptions = [item['style_description'] for item in response.data]
        logging.info(f"{len(style_descriptions)}件のスタイル記述を取得しました。")
        
        return QueryResponse(documents=style_descriptions)
    except Exception as e:
        logging.error(f"スタイルパレット取得(/get_style_palette)中にエラーが発生しました: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/update_character_states", response_model=CharacterStateUpdateResponse, tags=["Character Emotion"])
async def update_character_states(request: CharacterStateUpdateRequest):
    """
    会話の終了後、Botから送られてきたキャラクターの最新の感情状態でDBを更新する。
    """
    try:
        update_count = 0
        records_to_upsert = []
        for state in request.states:
            records_to_upsert.append({
                "user_id": state.user_id,
                "character_name": state.character_name,
                "mood": state.mood,
                "last_interaction_summary": state.last_interaction_summary
            })
        
        if records_to_upsert:
            supabase.table('character_states').upsert(records_to_upsert, on_conflict='user_id, character_name').execute()
            update_count = len(records_to_upsert)
        
        logging.info(f"{update_count}人のキャラクターの状態を更新しました。")
        return CharacterStateUpdateResponse(
            message=f"{update_count} characters' states have been successfully updated.",
            updated_count=update_count
        )
    except Exception as e:
        logging.error(f"キャラクター状態の更新(/update_character_states)中にエラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/get_character_states", response_model=CharacterStateQueryResponse, tags=["Character Emotion"])
async def get_character_states(user_id: str = "imazine"):
    """
    会話の開始前、Botがキャラクターの現在の感情状態をDBから取得する。
    """
    try:
        response = supabase.table('character_states').select("*").eq('user_id', user_id).execute()
        
        states = {}
        if response.data:
            for item in response.data:
                states[item['character_name']] = {
                    "mood": item.get('mood', 'ニュートラル'),
                    "last_interaction_summary": item.get('last_interaction_summary', '特筆すべきやり取りはなかった。')
                }
        
        if "みらい" not in states: states["みらい"] = {"mood": "ニュートラル", "last_interaction_summary": "まだ会話が始まっていません。"}
        if "へー子" not in states: states["へー子"] = {"mood": "ニュートラル", "last_interaction_summary": "まだ会話が始まっていません。"}
            
        logging.info(f"ユーザー({user_id})のキャラクター状態を取得しました。")
        return CharacterStateQueryResponse(states=states)
    except Exception as e:
        logging.error(f"キャラクター状態の取得(/get_character_states)中にエラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/log_concern", response_model=ConcernLogResponse, tags=["Character Care"])
async def log_concern(request: Concern):
    """Botが検知したimazineの心配事をDBに記録する"""
    try:
        response = supabase.table('concerns').insert({
            "user_id": request.user_id,
            "concern_text": request.concern_text
        }).execute()
        new_id = response.data[0]['id']
        logging.info(f"新しい心配事をID:{new_id}として記録しました。")
        return ConcernLogResponse(concern_id=new_id)
    except Exception as e:
        logging.error(f"心配事の記録(/log_concern)中にエラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_unresolved_concerns", response_model=UnresolvedConcernResponse, tags=["Character Care"])
async def get_unresolved_concerns(user_id: str = "imazine"):
    """未解決の心配事を取得する"""
    try:
        response = supabase.table('concerns').select("*").eq('user_id', user_id).eq('is_resolved', False).order('created_at').limit(5).execute()
        return UnresolvedConcernResponse(concerns=response.data)
    except Exception as e:
        logging.error(f"未解決の心配事の取得(/get_unresolved_concerns)中にエラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/resolve_concern", tags=["Character Care"])
async def resolve_concern(request: ResolveConcernRequest):
    """心配事が解決したことをマークする"""
    try:
        supabase.table('concerns').update({"is_resolved": True, "resolved_at": dt.datetime.now(dt.timezone.utc).isoformat()}).eq('id', request.concern_id).execute()
        logging.info(f"心配事ID:{request.concern_id}を解決済みにマークしました。")
        return {"status": "success", "message": "Concern marked as resolved."}
    except Exception as e:
        logging.error(f"心配事の解決(/resolve_concern)中にエラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
