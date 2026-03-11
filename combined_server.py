#!/usr/bin/env python3
"""
Shankh AR — All-in-One Server (port 5000)
==========================================
Handles everything in one process, one port, one ngrok tunnel:

  /                    → ar_new.html
  /generate_lipsync    → TTS + Rhubarb lip-sync
  /retrieve            → RAG FAISS semantic search
  /stock/price         → yfinance live stock price
  /stock/insights      → analyst recommendations (Google Sheets)
  /health              → status of all subsystems
  /audio/<id>          → serve generated audio files

Google Sheet columns (EXACT):
  Stock Name | Signal | confidence (out of 10) | weightedScore | Date

Run:
  python server.py

Expose via single ngrok tunnel:
  ngrok http 5000
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_compress import Compress
import os, subprocess, json, uuid, time, pickle, hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from openai import OpenAI

# ── RAG imports ───────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("WARNING: RAG disabled. Install: pip install sentence-transformers faiss-cpu numpy")

# ── Stock imports ─────────────────────────────────────────────────
try:
    import yfinance as yf
    import pandas as pd
    from datetime import datetime, timedelta
    STOCK_AVAILABLE = True
except ImportError:
    STOCK_AVAILABLE = False
    print("WARNING: Stock disabled. Install: pip install yfinance pandas")

# =============================================================================
# PATHS & CONFIG
# =============================================================================
BASE_DIR  = Path(__file__).parent.resolve()
AUDIO_DIR = BASE_DIR / "audio"
INDEX_DIR = BASE_DIR / "rag_service" / "index"
AUDIO_DIR.mkdir(exist_ok=True)

RHUBARB_PATH    = "/Users/allenpeter/Desktop/Rhubarb/Rhubarb-Lip-Sync-1.14.0-macOS/rhubarb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ✅ Google Sheet — column names confirmed from frontend code:
#    Stock Name | Signal | confidence | weightedScore | Date
INSIGHTS_SHEET_ID  = "13926Tv0c8xGj2vGDW1xqwTG9FY1cvlfpRLacM0fPLa0"
INSIGHTS_SHEET_URL = f"https://docs.google.com/spreadsheets/d/{INSIGHTS_SHEET_ID}/gviz/tq?tqx=out:csv"

# =============================================================================
# FLASK SETUP
# =============================================================================
app = Flask(__name__)
CORS(app,
     resources={r"/*": {"origins": "*"}},
     allow_headers=["Content-Type", "Authorization", "ngrok-skip-browser-warning"],
     methods=["GET", "POST", "OPTIONS"])
Compress(app)

STATIC_EXTS       = {'.js', '.wasm', '.glb', '.mind', '.jpg', '.png', '.html'}
NO_CACHE_PREFIXES = ('/generate_lipsync', '/retrieve', '/stock', '/health', '/audio')

@app.after_request
def set_cache(response):
    path = request.path
    if any(path.startswith(p) for p in NO_CACHE_PREFIXES):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma']        = 'no-cache'
        response.headers['Expires']       = '0'
    elif Path(path).suffix.lower() in STATIC_EXTS:
        response.headers['Cache-Control'] = 'public, max-age=86400'
    return response

# =============================================================================
# RAG STATE
# =============================================================================
rag_index       = None
rag_metadata    = None
rag_model       = None
rag_embed_cache = {}
rag_executor    = ThreadPoolExecutor(max_workers=2)

def load_rag():
    global rag_index, rag_metadata, rag_model
    if not RAG_AVAILABLE:
        return

    index_file    = INDEX_DIR / "index.faiss"
    metadata_file = INDEX_DIR / "metadata.pkl"

    if not index_file.exists():
        index_file = INDEX_DIR / "faiss_index.bin"

    if index_file.exists():
        rag_index = faiss.read_index(str(index_file))
        print(f"RAG: index loaded ({rag_index.ntotal} vectors)")
    else:
        print(f"RAG: no index found at {INDEX_DIR} — run ingest.py first")
        return

    if metadata_file.exists():
        with open(metadata_file, "rb") as f:
            raw = pickle.load(f)
        rag_metadata = raw.get("chunks", raw) if isinstance(raw, dict) else raw
        print(f"RAG: {len(rag_metadata)} chunks loaded")

    print(f"RAG: loading embedding model '{EMBEDDING_MODEL}'...")
    rag_model = SentenceTransformer(EMBEDDING_MODEL)
    print("RAG: ready")

# =============================================================================
# STOCK CACHE
# =============================================================================
_stock_cache    = {}   # symbol → (data, timestamp)
_insights_cache = {}   # "sheet" → (df, timestamp)

def _cache_get(cache, key, ttl_minutes):
    if key in cache:
        data, ts = cache[key]
        if (datetime.now() - ts).seconds < ttl_minutes * 60:
            return data
    return None

def _cache_set(cache, key, data):
    cache[key] = (data, datetime.now())

INDIAN_INDICES = {'nifty': '^NSEI', 'sensex': '^BSESN', 'banknifty': '^NSEBANK'}

def normalize_symbol(symbol):
    s = symbol.upper().strip()
    if s.lower() in INDIAN_INDICES:
        return INDIAN_INDICES[s.lower()]
    if s.endswith('.NS') or s.endswith('.BO'):
        return s
    return s + '.NS'

# =============================================================================
# STARTUP CHECKS
# =============================================================================
def check_dependencies():
    print("\n" + "="*55)
    print("  Shankh AR — All-in-One Server")
    print("="*55)
    print(f"  Base dir  : {BASE_DIR}")

    try:
        r = subprocess.run([RHUBARB_PATH, "--version"], capture_output=True, text=True, timeout=5)
        print(f"  Rhubarb   : {r.stdout.strip()}")
    except FileNotFoundError:
        print(f"  Rhubarb   : NOT FOUND at {RHUBARB_PATH}")
        exit(1)

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        print(f"  ffmpeg    : OK")
    except FileNotFoundError:
        print(f"  ffmpeg    : NOT FOUND — brew install ffmpeg")
        exit(1)

    required = ["ar_new.html", "targets.mind", "avatar_optimized.glb", "marker.jpg"]
    print("-"*55)
    for fname in required:
        fpath = BASE_DIR / fname
        status = f"{fpath.stat().st_size // 1024:>6} KB" if fpath.exists() else "MISSING ❌"
        print(f"  {fname:<30} {status}")

    print("-"*55)
    print(f"  RAG    : {'enabled' if RAG_AVAILABLE else 'disabled'}")
    print(f"  Stocks : {'enabled' if STOCK_AVAILABLE else 'disabled'}")
    print("="*55 + "\n")

check_dependencies()
load_rag()

# =============================================================================
# STATIC FILE ROUTES
# =============================================================================
@app.route('/')
def root():
    return send_from_directory(str(BASE_DIR), 'ar_new.html')

@app.route('/<path:filename>')
def serve_file(filename):
    if filename.endswith('.map'):
        return '', 204
    return send_from_directory(str(BASE_DIR), filename)

@app.route('/audio/<audio_id>')
def serve_audio(audio_id):
    return send_from_directory(str(AUDIO_DIR), f"{audio_id}.mp3")

# =============================================================================
# HEALTH
# =============================================================================
@app.route('/health')
def health():
    return jsonify({
        'status'     : 'ok',
        'rag_ready'  : rag_index is not None,
        'rag_chunks' : rag_index.ntotal if rag_index else 0,
        'stock_ready': STOCK_AVAILABLE,
        'base_dir'   : str(BASE_DIR),
    })

# =============================================================================
# LIP-SYNC
# =============================================================================
_lipsync_cache = {}
MAX_CACHE_SIZE = 50

@app.route('/generate_lipsync', methods=['POST'])
def generate_lipsync():
    start_time = time.time()
    data       = request.json
    text       = data.get('text', '').strip()
    api_key    = data.get('api_key', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    text_hash = hashlib.md5(text.encode()).hexdigest()
    if text_hash in _lipsync_cache:
        cached_id   = _lipsync_cache[text_hash]
        cached_mp3  = AUDIO_DIR / f"{cached_id}.mp3"
        cached_json = AUDIO_DIR / f"{cached_id}.json"
        if cached_mp3.exists() and cached_json.exists():
            with open(cached_json, 'r') as f:
                lipsync_data = json.load(f)
            print(f"[CACHE HIT] {cached_id}")
            return jsonify({'audio_url': f'/audio/{cached_id}', 'lipsync': lipsync_data})
        else:
            del _lipsync_cache[text_hash]

    audio_id  = str(uuid.uuid4())[:8]
    mp3_path  = AUDIO_DIR / f"{audio_id}.mp3"
    wav_path  = AUDIO_DIR / f"{audio_id}.wav"
    json_path = AUDIO_DIR / f"{audio_id}.json"

    print(f"\n[{audio_id}] '{text[:50]}...'")

    try:
        t = time.time()
        client = OpenAI(api_key=api_key)
        resp   = client.audio.speech.create(model="tts-1", voice="alloy", input=text, speed=1.1)
        resp.stream_to_file(str(mp3_path))
        print(f"[{audio_id}] TTS {time.time()-t:.2f}s")

        t = time.time()
        subprocess.run([
            'ffmpeg', '-i', str(mp3_path),
            '-ar', '16000', '-ac', '1', '-y', '-loglevel', 'quiet', str(wav_path)
        ], check=True, timeout=10)
        print(f"[{audio_id}] ffmpeg {time.time()-t:.2f}s")

        t = time.time()
        subprocess.run([
            RHUBARB_PATH, '-f', 'json', '-o', str(json_path),
            '--recognizer', 'phonetic', '--machineReadable', str(wav_path)
        ], capture_output=True, text=True, timeout=30, check=True)
        print(f"[{audio_id}] Rhubarb {time.time()-t:.2f}s")

        with open(json_path, 'r') as f:
            lipsync_data = json.load(f)

        if len(_lipsync_cache) >= MAX_CACHE_SIZE:
            oldest = next(iter(_lipsync_cache))
            old_id = _lipsync_cache.pop(oldest)
            (AUDIO_DIR / f"{old_id}.mp3").unlink(missing_ok=True)
            (AUDIO_DIR / f"{old_id}.json").unlink(missing_ok=True)
        _lipsync_cache[text_hash] = audio_id

        print(f"[{audio_id}] TOTAL {time.time()-start_time:.2f}s")
        return jsonify({'audio_url': f'/audio/{audio_id}', 'lipsync': lipsync_data})

    except Exception as e:
        print(f"[{audio_id}] ERROR: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        wav_path.unlink(missing_ok=True)

# =============================================================================
# RAG
# =============================================================================
@app.route('/retrieve', methods=['POST'])
def retrieve():
    if not RAG_AVAILABLE or rag_index is None or rag_model is None:
        return jsonify({'results': [], 'num_results': 0, 'rag_ready': False}), 200

    data      = request.json
    query     = data.get('query', '').strip()
    k         = int(data.get('k', 5))
    threshold = float(data.get('threshold', 0.0))

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    try:
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in rag_embed_cache:
            embedding = rag_embed_cache[cache_key]
        else:
            future    = rag_executor.submit(rag_model.encode, [query])
            embedding = future.result(timeout=5)
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            if len(rag_embed_cache) > 200:
                del rag_embed_cache[next(iter(rag_embed_cache))]
            rag_embed_cache[cache_key] = embedding

        distances, indices = rag_index.search(embedding.astype(np.float32), k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or float(dist) < threshold:
                continue
            if rag_metadata and idx < len(rag_metadata):
                meta = rag_metadata[idx]
                results.append({
                    'filename': meta.get('filename', 'unknown'),
                    'page_num': meta.get('page_num', 0),
                    'text'    : meta.get('text', ''),
                    'excerpt' : meta.get('text', '')[:200],
                    'score'   : float(dist),
                })

        print(f"RAG: '{query[:40]}' → {len(results)} results")
        return jsonify({'results': results, 'num_results': len(results), 'query': query})

    except Exception as e:
        print(f"RAG retrieve error: {e}")
        return jsonify({'error': str(e)}), 500

# =============================================================================
# STOCK PRICE
# =============================================================================
@app.route('/stock/price', methods=['POST'])
def stock_price():
    if not STOCK_AVAILABLE:
        return jsonify({'error': 'yfinance not installed'}), 503

    symbol = request.json.get('symbol', '').strip()
    if not symbol:
        return jsonify({'error': 'No symbol provided'}), 400

    cached = _cache_get(_stock_cache, symbol.upper(), ttl_minutes=5)
    if cached:
        return jsonify(cached)

    try:
        ns     = normalize_symbol(symbol)
        ticker = yf.Ticker(ns)
        info   = ticker.info

        price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
        if not price:
            return jsonify({'error': f'Price not found for {symbol}'}), 404

        prev = info.get('previousClose')
        data = {
            'symbol'          : symbol.upper(),
            'normalized_symbol': ns,
            'company_name'    : info.get('longName', info.get('shortName', symbol)),
            'current_price'   : round(price, 2),
            'currency'        : info.get('currency', 'INR'),
            'previous_close'  : prev,
            'open'            : info.get('open'),
            'day_high'        : info.get('dayHigh'),
            'day_low'         : info.get('dayLow'),
            'volume'          : info.get('volume'),
            'market_cap'      : info.get('marketCap'),
            'pe_ratio'        : info.get('trailingPE'),
            'change'          : round(price - prev, 2)            if prev else None,
            'change_percent'  : round((price - prev)/prev*100, 2) if prev else None,
            'timestamp'       : datetime.now().isoformat(),
        }
        _cache_set(_stock_cache, symbol.upper(), data)
        print(f"Stock: {symbol} = Rs {price}")
        return jsonify(data)

    except Exception as e:
        print(f"Stock error for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

# =============================================================================
# STOCK INSIGHTS — Google Sheets
# ✅ FIXED: using correct column names from the sheet:
#    Stock Name | Signal | confidence | weightedScore | Date
# =============================================================================
@app.route('/stock/insights', methods=['POST'])
def stock_insights():
    if not STOCK_AVAILABLE:
        return jsonify({'error': 'pandas not installed'}), 503

    stock_name = request.json.get('stock_name', '').strip().lower()
    if not stock_name:
        return jsonify({'error': 'No stock_name provided'}), 400

    try:
        # Fetch & cache the sheet (15-min TTL)
        df = _cache_get(_insights_cache, 'sheet', ttl_minutes=15)
        if df is None:
            print(f"Insights: fetching Google Sheet...")
            df = pd.read_csv(INSIGHTS_SHEET_URL)
            # Strip whitespace from all string columns
            df.columns = df.columns.str.strip()
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            _cache_set(_insights_cache, 'sheet', df)
            print(f"Insights: loaded {len(df)} rows, columns: {df.columns.tolist()}")

        # Case-insensitive match on 'Stock Name'
        rows = df[df['Stock Name'].str.lower() == stock_name]

        if rows.empty:
            print(f"Insights: no match for '{stock_name}' — available: {df['Stock Name'].str.lower().unique().tolist()}")
            return jsonify({'error': f'No insights for {stock_name}'}), 404

        # Use the most recent row
        row = rows.iloc[-1]

        # ✅ CORRECT column names from actual Google Sheet:
        # Date | Stock Name | ticker | Signal | FMP Recommendation | source | Timestamp
        signal          = int(row['Signal'])
        fmp_rec         = str(row['FMP Recommendation']).strip()  # e.g. "BUY", "SELL", "HOLD"

        # Signal int → label (our model's signal, separate from FMP)
        signal_label = 'BUY' if signal == 1 else 'SELL' if signal == -1 else 'HOLD'

        # Build rich recommendation text for OpenAI to reason about
        # Combines our signal + FMP analyst recommendation
        if signal_label == fmp_rec:
            agreement = f"Both our model and FMP analysts agree: {signal_label}."
        else:
            agreement = f"Our model says {signal_label} while FMP analysts recommend {fmp_rec}."

        recommendation_text = (
            f"{agreement} "
            f"Stock: {row['Stock Name']} ({row.get('ticker', '')}). "
            f"Data as of {row['Date']}. "
            f"Based on this, give a direct investment opinion."
        )

        result = {
            'stock_name'      : str(row['Stock Name']),
            'ticker'          : str(row.get('ticker', '')),
            'signal'          : signal,
            'signal_label'    : signal_label,       # from our model: BUY/SELL/HOLD
            'fmp_rec'         : fmp_rec,            # from FMP analysts: BUY/SELL/HOLD
            'agreement'       : agreement,
            'recommendation'  : recommendation_text,
            'date'            : str(row['Date']),
            'timestamp'       : datetime.now().isoformat(),
        }

        print(f"Insights: {stock_name} → model:{signal_label} fmp:{fmp_rec}")
        return jsonify(result)

    except KeyError as e:
        # Column name mismatch — log actual columns to help debug
        print(f"Insights KeyError: {e}")
        print(f"Actual columns: {df.columns.tolist() if df is not None else 'df not loaded'}")
        return jsonify({'error': f'Column not found: {e}. Check server logs for actual columns.'}), 500

    except Exception as e:
        print(f"Insights error for {stock_name}: {e}")
        return jsonify({'error': str(e)}), 500

# =============================================================================
# STOCK SEARCH
# =============================================================================
@app.route('/stock/search', methods=['POST'])
def stock_search():
    query  = request.json.get('query', '').lower()
    STOCKS = {
        'RELIANCE':'Reliance Industries','TCS':'Tata Consultancy Services',
        'HDFCBANK':'HDFC Bank','INFY':'Infosys','ICICIBANK':'ICICI Bank',
        'HINDUNILVR':'Hindustan Unilever','ITC':'ITC Limited','SBIN':'State Bank of India',
        'BHARTIARTL':'Bharti Airtel','KOTAKBANK':'Kotak Mahindra Bank',
        'WIPRO':'Wipro','BAJFINANCE':'Bajaj Finance','ASIANPAINT':'Asian Paints',
        'MARUTI':'Maruti Suzuki','AXISBANK':'Axis Bank','LT':'Larsen & Toubro',
        'TITAN':'Titan Company','SUNPHARMA':'Sun Pharmaceutical',
        'ULTRACEMCO':'UltraTech Cement','NESTLEIND':'Nestle India',
    }
    results = [
        {'symbol': s, 'name': n, 'exchange': 'NSE'}
        for s, n in STOCKS.items()
        if query in s.lower() or query in n.lower()
    ]
    return jsonify({'results': results[:10]})

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print(f"  Open     → http://localhost:5000")
    print(f"  Health   → http://localhost:5000/health")
    print(f"  Test insights → curl -X POST http://localhost:5000/stock/insights "
          f"-H 'Content-Type: application/json' -d '{{\"stock_name\":\"reliance\"}}'")
    print()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)