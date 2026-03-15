#!/usr/bin/env python3
"""
Shankh AR — All-in-One Server (port 5000)
==========================================
  /                    → ar_new.html  (or index.html)
  /generate_lipsync    → OpenAI TTS + Rhubarb lip-sync  (English fallback only)
  /retrieve            → RAG FAISS semantic search
  /stock/price         → yfinance live stock price
  /stock/insights      → analyst recommendations (Google Sheets)
  /proxy/chat          → OpenAI GPT proxy  (avoids browser CORS block)
  /proxy/detect_lang   → OpenAI language detection proxy
  /transcribe          → OpenAI Whisper STT proxy  (avoids browser CORS + far better than Web Speech API)
  /health              → status of all subsystems
  /audio/<id>          → serve generated audio files

Run:
  export OPENAI_API_KEY="sk-proj-your-key-here"
  python combined_server.py

Expose via ngrok:
  ngrok http 5000
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_compress import Compress
import os, subprocess, json, uuid, time, pickle, hashlib, threading
import requests as req_lib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from openai import OpenAI

# ── RAG imports ───────────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("⚠️  RAG disabled. Run: pip install sentence-transformers faiss-cpu numpy")

# ── Stock imports ─────────────────────────────────────────────────────────────
try:
    import yfinance as yf
    import pandas as pd
    from datetime import datetime, timedelta
    STOCK_AVAILABLE = True
except ImportError:
    STOCK_AVAILABLE = False
    print("⚠️  Stock disabled. Run: pip install yfinance pandas")

# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR  = Path(__file__).parent.resolve()
AUDIO_DIR = BASE_DIR / "audio"
INDEX_DIR = BASE_DIR / "rag_service" / "index"
AUDIO_DIR.mkdir(exist_ok=True)

RHUBARB_PATH    = "/Users/allenpeter/Desktop/Rhubarb/Rhubarb-Lip-Sync-1.14.0-macOS/rhubarb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ✅ Read API key from environment variable ONLY — never hardcode in source
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("⚠️  WARNING: OPENAI_API_KEY not set. Run: export OPENAI_API_KEY='sk-proj-...'")

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

NO_CACHE_PREFIXES = ('/generate_lipsync', '/retrieve', '/stock', '/health', '/audio', '/proxy', '/transcribe')
STATIC_EXTS       = {'.js', '.wasm', '.glb', '.mind', '.jpg', '.png', '.html'}

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
# RAG
# =============================================================================
rag_index       = None
rag_metadata    = None
rag_model       = None
rag_embed_cache = {}
rag_executor    = ThreadPoolExecutor(max_workers=4)

def load_rag():
    global rag_index, rag_metadata, rag_model
    if not RAG_AVAILABLE:
        return

    index_file    = INDEX_DIR / "faiss_index.bin"
    if not index_file.exists():
        index_file = INDEX_DIR / "index.faiss"
    metadata_file = INDEX_DIR / "metadata.pkl"

    if not index_file.exists():
        print(f"⚠️  RAG: no index at {INDEX_DIR} — run ingest.py first")
        return

    try:
        rag_index = faiss.read_index(str(index_file))
        print(f"✅ RAG: index loaded ({rag_index.ntotal} vectors, dim={rag_index.d})")
    except Exception as e:
        print(f"❌ RAG index load failed: {e}")
        return

    if metadata_file.exists():
        with open(metadata_file, "rb") as f:
            raw = pickle.load(f)
        rag_metadata = raw.get("chunks", raw) if isinstance(raw, dict) else raw
        print(f"✅ RAG: {len(rag_metadata)} chunks")

    dim = rag_index.d
    selected_model = {384: 'all-MiniLM-L6-v2', 768: 'all-mpnet-base-v2'}.get(dim, EMBEDDING_MODEL)
    if selected_model != EMBEDDING_MODEL:
        print(f"⚠️  Index dim={dim} — using '{selected_model}'")

    print(f"⏳ RAG: loading embedding model '{selected_model}'...")
    try:
        rag_model = SentenceTransformer(selected_model)
        print("✅ RAG: model loaded")
    except Exception as e:
        print(f"❌ RAG model load failed: {e}")
        rag_index = None
        return

    def _warmup():
        try:
            print("🔥 Warming up embedding model...")
            _ = rag_model.encode(["warmup query"], show_progress_bar=False)
            print("✅ Embedding model warmed up — RAG ready")
        except Exception as e:
            print(f"⚠️  Embedding warmup failed (non-fatal): {e}")
    threading.Thread(target=_warmup, daemon=True).start()

# =============================================================================
# STOCK CACHE
# =============================================================================
_stock_cache    = {}
_insights_cache = {}

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
    print("  Shankh AR — Combined Server")
    print("="*55)
    print(f"  Base dir : {BASE_DIR}")

    try:
        r = subprocess.run([RHUBARB_PATH, "--version"], capture_output=True, text=True, timeout=5)
        print(f"  Rhubarb  : {r.stdout.strip()}")
    except FileNotFoundError:
        print(f"  Rhubarb  : ❌ NOT FOUND at {RHUBARB_PATH}")
        exit(1)

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        print(f"  ffmpeg   : ✅")
    except FileNotFoundError:
        print(f"  ffmpeg   : ❌ — run: brew install ffmpeg")
        exit(1)

    print(f"  OpenAI   : {'✅ key set' if OPENAI_API_KEY else '❌ NOT SET — export OPENAI_API_KEY=...'}")

    required = ["ar_new.html", "targets.mind", "avatar_hq.glb", "marker.jpg"]
    print("-"*55)
    for fname in required:
        fpath = BASE_DIR / fname
        if fpath.exists():
            print(f"  {fname:<30} {fpath.stat().st_size // 1024:>6} KB")
        else:
            if fname == "ar_new.html" and (BASE_DIR / "index.html").exists():
                print(f"  {fname:<30} (serving index.html instead)")
                continue
            print(f"  {fname:<30} ❌ MISSING")

    print("-"*55)
    print(f"  RAG      : {'enabled' if RAG_AVAILABLE else 'disabled'}")
    print(f"  Stocks   : {'enabled' if STOCK_AVAILABLE else 'disabled'}")
    print("="*55 + "\n")

check_dependencies()
load_rag()

# =============================================================================
# STATIC ROUTES
# =============================================================================
@app.route('/')
def root():
    for fname in ['ar_new.html', 'index.html']:
        if (BASE_DIR / fname).exists():
            return send_from_directory(str(BASE_DIR), fname)
    return "No HTML file found", 404

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
        'openai_key' : bool(OPENAI_API_KEY),
        'base_dir'   : str(BASE_DIR),
    })

# =============================================================================
# OPENAI PROXY ROUTES
# Browser → Flask → OpenAI  (fixes CORS block on direct browser calls)
# =============================================================================
@app.route('/proxy/chat', methods=['POST'])
def proxy_chat():
    """Proxy for GPT chat completions — avoids browser CORS restrictions."""
    if not OPENAI_API_KEY:
        return jsonify({'error': 'OPENAI_API_KEY not set on server'}), 500

    data = request.json or {}
    try:
        r = req_lib.post(
            'https://api.openai.com/v1/chat/completions',
            headers={
                'Content-Type':  'application/json',
                'Authorization': f'Bearer {OPENAI_API_KEY}',
            },
            json=data,
            timeout=30
        )
        return jsonify(r.json()), r.status_code
    except Exception as e:
        print(f"proxy/chat error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/proxy/detect_lang', methods=['POST'])
def proxy_detect_lang():
    """Proxy for language detection via GPT — avoids browser CORS restrictions."""
    if not OPENAI_API_KEY:
        return jsonify({'error': 'OPENAI_API_KEY not set on server'}), 500

    text = (request.json or {}).get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        r = req_lib.post(
            'https://api.openai.com/v1/chat/completions',
            headers={
                'Content-Type':  'application/json',
                'Authorization': f'Bearer {OPENAI_API_KEY}',
            },
            json={
                'model':       'gpt-4o-mini',
                'max_tokens':  5,
                'temperature': 0,
                'messages': [
                    {
                        'role':    'system',
                        'content': 'Detect language. Reply ONLY with the ISO 639-1 code: hi,ta,te,kn,ml,bn,mr,gu,pa,or,ur,as,en,mni,sat. Nothing else.'
                    },
                    {'role': 'user', 'content': text}
                ]
            },
            timeout=10
        )
        return jsonify(r.json()), r.status_code
    except Exception as e:
        print(f"proxy/detect_lang error: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# ✅ NEW: WHISPER TRANSCRIPTION PROXY
# Replaces browser Web Speech API — far more accurate for Indian languages/accents.
# Accepts: multipart/form-data with 'audio' file + optional 'lang' field.
# Returns: { text: "...", detected_language: "..." }
# =============================================================================
@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Transcribe audio using OpenAI Whisper.
    Accepts multipart form with:
      - audio: audio blob (webm/ogg/wav/mp4 — whatever MediaRecorder produces)
      - lang:  optional ISO 639-1 hint (e.g. 'hi', 'ta') — omit for auto-detect
    Returns: { text, detected_language }
    """
    if not OPENAI_API_KEY:
        return jsonify({'error': 'OPENAI_API_KEY not set on server'}), 500

    audio_file = request.files.get('audio')
    if not audio_file:
        return jsonify({'error': 'No audio file provided'}), 400

    lang_hint = request.form.get('lang', '').strip() or None

    # Save incoming blob to a temp file with an appropriate extension
    orig_name  = audio_file.filename or 'audio.webm'
    ext        = Path(orig_name).suffix or '.webm'
    tmp_path   = AUDIO_DIR / f"whisper_{uuid.uuid4().hex[:8]}{ext}"

    try:
        audio_file.save(str(tmp_path))
        file_size = tmp_path.stat().st_size
        if file_size < 1000:
            return jsonify({'error': 'Audio too short or empty', 'bytes': file_size}), 400

        client = OpenAI(api_key=OPENAI_API_KEY)

        with open(tmp_path, 'rb') as f:
            kwargs = {
                'model': 'whisper-1',
                'file' : f,
                'response_format': 'verbose_json',   # gives us detected_language too
            }
            # Only pass language hint for non-English to help Whisper
            if lang_hint and lang_hint != 'en':
                kwargs['language'] = lang_hint

            result = client.audio.transcriptions.create(**kwargs)

        text      = (result.text or '').strip()
        detected  = getattr(result, 'language', lang_hint or 'en')

        print(f"Whisper: lang={detected}, text='{text[:60]}'")
        return jsonify({'text': text, 'detected_language': detected})

    except Exception as e:
        print(f"Whisper error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        tmp_path.unlink(missing_ok=True)


# =============================================================================
# LIP-SYNC
# =============================================================================
_lipsync_cache = {}
MAX_CACHE_SIZE = 50

@app.route('/generate_lipsync', methods=['POST'])
def generate_lipsync():
    start_time = time.time()
    data       = request.json or {}
    text       = data.get('text', '').strip()

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
            return jsonify({'audio_url': f'/audio/{cached_id}', 'lipsync': lipsync_data})
        del _lipsync_cache[text_hash]

    audio_id  = str(uuid.uuid4())[:8]
    mp3_path  = AUDIO_DIR / f"{audio_id}.mp3"
    wav_path  = AUDIO_DIR / f"{audio_id}.wav"
    json_path = AUDIO_DIR / f"{audio_id}.json"

    print(f"\n[{audio_id}] lipsync: '{text[:60]}'")

    try:
        t = time.time()
        client = OpenAI(api_key=OPENAI_API_KEY)
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
# RAG RETRIEVE
# =============================================================================
@app.route('/retrieve', methods=['POST'])
def retrieve():
    if not RAG_AVAILABLE or rag_index is None or rag_model is None:
        return jsonify({'results': [], 'num_results': 0, 'rag_ready': False}), 200

    data      = request.json or {}
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
            future = rag_executor.submit(rag_model.encode, [query])
            try:
                embedding = future.result(timeout=30)
            except TimeoutError:
                print(f"⚠️  RAG: embedding timed out for query '{query[:40]}' — returning empty")
                return jsonify({'results': [], 'num_results': 0, 'timeout': True}), 200

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
        import traceback
        print(f"RAG error: {e}")
        traceback.print_exc()
        return jsonify({'results': [], 'num_results': 0, 'error': str(e)}), 200

# =============================================================================
# STOCK PRICE
# =============================================================================
@app.route('/stock/price', methods=['POST'])
def stock_price():
    if not STOCK_AVAILABLE:
        return jsonify({'error': 'yfinance not installed'}), 503

    symbol = (request.json or {}).get('symbol', '').strip()
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
            'symbol'           : symbol.upper(),
            'normalized_symbol': ns,
            'company_name'     : info.get('longName', info.get('shortName', symbol)),
            'current_price'    : round(price, 2),
            'currency'         : info.get('currency', 'INR'),
            'previous_close'   : prev,
            'open'             : info.get('open'),
            'day_high'         : info.get('dayHigh'),
            'day_low'          : info.get('dayLow'),
            'volume'           : info.get('volume'),
            'market_cap'       : info.get('marketCap'),
            'pe_ratio'         : info.get('trailingPE'),
            'change'           : round(price - prev, 2)            if prev else None,
            'change_percent'   : round((price - prev)/prev*100, 2) if prev else None,
            'timestamp'        : datetime.now().isoformat(),
        }
        _cache_set(_stock_cache, symbol.upper(), data)
        print(f"Stock: {symbol} = Rs {price}")
        return jsonify(data)

    except Exception as e:
        print(f"Stock error [{symbol}]: {e}")
        return jsonify({'error': str(e)}), 500

# =============================================================================
# STOCK INSIGHTS — Google Sheets
# =============================================================================
@app.route('/stock/insights', methods=['POST'])
def stock_insights():
    if not STOCK_AVAILABLE:
        return jsonify({'error': 'pandas not installed'}), 503

    stock_name = (request.json or {}).get('stock_name', '').strip().lower()
    if not stock_name:
        return jsonify({'error': 'No stock_name provided'}), 400

    try:
        df = _cache_get(_insights_cache, 'sheet', ttl_minutes=15)
        if df is None:
            print("Insights: fetching Google Sheet...")
            df = pd.read_csv(INSIGHTS_SHEET_URL)
            df.columns = df.columns.str.strip()
            df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
            _cache_set(_insights_cache, 'sheet', df)
            print(f"Insights: {len(df)} rows")

        rows = df[df['Stock Name'].str.lower() == stock_name]
        if rows.empty:
            return jsonify({'error': f'No insights for {stock_name}'}), 404

        row          = rows.iloc[-1]
        signal       = int(row['Signal'])
        fmp_rec      = str(row['FMP Recommendation']).strip()
        signal_label = 'BUY' if signal == 1 else 'SELL' if signal == -1 else 'HOLD'
        agreement    = (f"Both agree: {signal_label}." if signal_label == fmp_rec
                        else f"Model says {signal_label}, FMP says {fmp_rec}.")

        result = {
            'stock_name'  : str(row['Stock Name']),
            'ticker'      : str(row.get('ticker', '')),
            'signal'      : signal,
            'signal_label': signal_label,
            'fmp_rec'     : fmp_rec,
            'agreement'   : agreement,
            'date'        : str(row['Date']),
            'timestamp'   : datetime.now().isoformat(),
        }
        print(f"Insights: {stock_name} → {signal_label}")
        return jsonify(result)

    except KeyError as e:
        return jsonify({'error': f'Column not found: {e}'}), 500
    except Exception as e:
        print(f"Insights error [{stock_name}]: {e}")
        return jsonify({'error': str(e)}), 500

# =============================================================================
# STOCK SEARCH
# =============================================================================
@app.route('/stock/search', methods=['POST'])
def stock_search():
    query = (request.json or {}).get('query', '').lower()
    STOCKS = {
        'RELIANCE':'Reliance Industries','TCS':'Tata Consultancy Services',
        'HDFCBANK':'HDFC Bank','INFY':'Infosys','ICICIBANK':'ICICI Bank',
        'HINDUNILVR':'Hindustan Unilever','ITC':'ITC Limited','SBIN':'State Bank of India',
        'BHARTIARTL':'Bharti Airtel','KOTAKBANK':'Kotak Mahindra Bank',
        'WIPRO':'Wipro','BAJFINANCE':'Bajaj Finance','ASIANPAINT':'Asian Paints',
        'MARUTI':'Maruti Suzuki','AXISBANK':'Axis Bank','LT':'Larsen & Toubro',
        'TITAN':'Titan Company','SUNPHARMA':'Sun Pharmaceutical',
    }
    results = [{'symbol':s,'name':n,'exchange':'NSE'} for s,n in STOCKS.items()
               if query in s.lower() or query in n.lower()]
    return jsonify({'results': results[:10]})

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print(f"  Open   → http://localhost:5000")
    print(f"  Health → http://localhost:5000/health\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
