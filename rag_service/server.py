import os
import time
from typing import List, Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import Field, BaseModel

try:
    from pydantic_settings import BaseSettings
except ImportError:
    try:
        from pydantic.v1 import BaseSettings
    except ImportError:
        from pydantic import BaseSettings

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Optional: Language detection
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# Stock service
try:
    from stock_service import StockPriceService, StockInsightsService
    STOCK_SERVICE_AVAILABLE = True
except ImportError:
    STOCK_SERVICE_AVAILABLE = False
    print("Warning: stock_service not available. Install yfinance and pandas to enable stock features.")

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Server configuration from environment variables"""
    embedding_model: str = Field(
        default="paraphrase-multilingual-mpnet-base-v2",
        env="EMBEDDING_MODEL"
    )
    index_path: str = Field(default="./index", env="INDEX_PATH")
    whisper_model: str = Field(default="base", env="WHISPER_MODEL")

    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()


class ServerState:
    """Global server state"""
    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.metadata: Optional[Dict] = None
        self.model: Optional[SentenceTransformer] = None
        self.whisper_model: Optional[Any] = None
        self.start_time: datetime = datetime.now()
        self.ready: bool = False


state = ServerState()

# Initialize FastAPI
app = FastAPI(title="Shankh.ai RAG Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranscriptionResponse(BaseModel):
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    segments: Optional[List[Dict[str, Any]]] = None


def load_embedding_model():
    print(f"Loading embedding model: {settings.embedding_model}...")
    try:
        state.model = SentenceTransformer(settings.embedding_model)
        print("✓ Embedding model loaded")
    except Exception as e:
        print(f"Error loading embedding model: {e}")


def load_index_and_metadata():
    print(f"Loading FAISS index from: {settings.index_path}...")
    try:
        index_file = os.path.join(settings.index_path, "index.faiss")
        if os.path.exists(index_file):
            state.index = faiss.read_index(index_file)
            print("✓ Index loaded")
        else:
            print("⚠ Index file not found.")

        metadata_file = os.path.join(settings.index_path, "metadata.pkl")
        if os.path.exists(metadata_file):
            import pickle
            with open(metadata_file, "rb") as f:
                state.metadata = pickle.load(f)
            print("✓ Metadata loaded")
    except Exception as e:
        print(f"Error loading index: {e}")


def load_whisper_model():
    if not settings.whisper_model:
        return
    print(f"Loading Whisper model: {settings.whisper_model}...")
    try:
        import whisper
        state.whisper_model = whisper.load_model(settings.whisper_model)
        print("✓ Whisper loaded")
    except Exception as e:
        print(f"Warning: Whisper load failed: {e}")


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    print("=" * 70)
    print("  Shankh.ai RAG Service Starting...")
    print("=" * 70)

    try:
        load_embedding_model()
        load_index_and_metadata()
        load_whisper_model()

        state.ready = True
        print("=" * 70)
        print("  ✓ RAG Service Ready!")
        print("=" * 70)

    except Exception as e:
        print(f"✗ Startup failed: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(audio: UploadFile = File(...), language: Optional[str] = None):
    """
    Transcribe audio using Whisper (local).
    Supports all languages; pass language hint for accuracy improvement.
    """
    if not state.whisper_model:
        raise HTTPException(
            status_code=501,
            detail="Whisper STT model not loaded. Check WHISPER_MODEL env var."
        )

    import tempfile

    try:
        suffix = ".wav"
        if audio.filename:
            ext = os.path.splitext(audio.filename)[1]
            if ext:
                suffix = ext

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_path = temp_file.name

        print(f"[STT] Transcribing: {audio.filename} ({len(content)} bytes)")
        start_time = time.time()

        # Whisper transcription
        transcribe_kwargs = {}
        if language and language != "auto":
            transcribe_kwargs["language"] = language

        result = state.whisper_model.transcribe(temp_path, **transcribe_kwargs)

        elapsed = time.time() - start_time

        if os.path.exists(temp_path):
            os.unlink(temp_path)

        print(f"[STT] ✓ Whisper completed in {elapsed:.2f}s")

        return TranscriptionResponse(
            text=result["text"].strip(),
            language=result.get("language", language or "unknown"),
            confidence=0.85,
            segments=[]
        )

    except HTTPException:
        raise
    except Exception as e:
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


class RetrieveRequest(BaseModel):
    query: str
    k: int = 5
    threshold: float = 0.0
    lang_hint: Optional[str] = None


class RetrieveResult(BaseModel):
    filename: str
    page_num: int
    text: str
    excerpt: str
    score: float


@app.post("/retrieve")
async def retrieve(request: RetrieveRequest):
    """Retrieve relevant document chunks using FAISS semantic search"""
    if not state.model or not state.index:
        raise HTTPException(status_code=503, detail="RAG index not loaded")

    try:
        # Encode query
        query_embedding = state.model.encode([request.query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        # Search FAISS
        distances, indices = state.index.search(
            query_embedding.astype(np.float32), request.k
        )

        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            score = float(distance)

            # Apply threshold filter
            if score < request.threshold:
                continue

            if state.metadata and idx < len(state.metadata):
                meta = state.metadata[idx]
                excerpt = meta.get("text", "")[:200]
                results.append({
                    "filename": meta.get("filename", "unknown"),
                    "page_num": meta.get("page_num", 0),
                    "text": meta.get("text", ""),
                    "excerpt": excerpt,
                    "score": score,
                })

        return {
            "results": results,
            "num_results": len(results),
            "query": request.query,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")


@app.get("/health")
async def health():
    """Simple health check"""
    return {"status": "healthy", "ready": state.ready}


@app.get("/status")
async def status():
    """Detailed service status"""
    return {
        "service": "RAG Retrieval Service",
        "ready": state.ready,
        "uptime_seconds": (datetime.now() - state.start_time).seconds,
        "index_loaded": state.index is not None,
        "embedding_model": settings.embedding_model,
        "whisper_model": settings.whisper_model if state.whisper_model else None,
        "num_chunks": state.index.ntotal if state.index else 0,
    }


# =============================================================================
# Stock Price Endpoints (yfinance integration)
# =============================================================================

if STOCK_SERVICE_AVAILABLE:
    stock_service = StockPriceService()
    insights_service = StockInsightsService()

    class StockPriceRequest(BaseModel):
        """Request schema for stock price endpoint"""
        symbol: str = Field(..., description="Stock symbol (e.g., RELIANCE, TCS, INFY)")

    class MultipleStocksRequest(BaseModel):
        """Request schema for multiple stocks"""
        symbols: List[str] = Field(..., description="List of stock symbols")

    class StockSearchRequest(BaseModel):
        """Request schema for stock search"""
        query: str = Field(..., description="Search query (company name or symbol)")

    @app.post("/stock/price")
    async def get_stock_price(request: StockPriceRequest):
        """Get current stock price for Indian market"""
        data = stock_service.get_stock_price(request.symbol)
        if not data:
            raise HTTPException(status_code=404, detail=f"Stock not found: {request.symbol}")
        return data

    @app.post("/stock/multiple")
    async def get_multiple_stocks(request: MultipleStocksRequest):
        """Get prices for multiple stocks"""
        return stock_service.get_multiple_stocks(request.symbols)

    @app.post("/stock/search")
    async def search_stocks(request: StockSearchRequest):
        """Search for stocks by name or symbol"""
        return {"results": stock_service.search_stock(request.query)}

    class StockInsightsRequest(BaseModel):
        """Request schema for stock insights endpoint"""
        stock_name: str = Field(..., description="Stock name (e.g., reliance, tcs, infosys)")

    @app.post("/stock/insights")
    async def get_stock_insights(request: StockInsightsRequest):
        """Get investment insights for a stock from Google Sheets"""
        data = insights_service.get_stock_insights(request.stock_name)
        if not data:
            raise HTTPException(status_code=404, detail=f"Insights not found for: {request.stock_name}")
        return data

    @app.get("/stock/insights/all")
    async def get_all_stock_insights():
        """Get insights for all stocks"""
        return {"insights": insights_service.get_all_insights()}


if __name__ == "__main__":
    import uvicorn

    print(f"Starting server on {settings.host}:{settings.port}")
    uvicorn.run(
        "server:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info"
    )
