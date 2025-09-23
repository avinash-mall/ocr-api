#!/usr/bin/env python3
"""
FastAPI + EasyOCR endpoint optimized for low-quality English/Arabic scans (forms, certificates).

Key improvements:
- Accuracy-first defaults (beamsearch decoder, rotation checks, per-region contrast retry).
- Three preprocessing variants:
  1) COLOR: white-balanced, denoised, CLAHE on L*, slight sharpen (helps detector)
  2) GRAY: illumination-corrected, CLAHE, deskewed, slight sharpen (helps faint text)
  3) BINARIZED: adaptive threshold + light morphology (for messy backgrounds)
  The service runs COLOR first, falls back to GRAY, then BINARIZED if confidence is low.
- Upscaling enabled by default for tiny text; sizes snapped to multiples of 32 for detector stability.
- Optional allowlist/blocklist for targeted fields (e.g., digits-only).
- Reuses a single Reader with thread lock (EasyOCR not guaranteed thread-safe).

Env toggles:
- OCR_USE_GPU = auto|true|false   (default: auto → torch.cuda.is_available())
- EASYOCR_MODEL_DIR                (default: ./easyocr_models)
"""

# Environment setup for PyTorch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import io
import math
import base64
import requests
import json
from typing import List, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from PIL import Image, ImageOps, ImageCms
from threading import Lock

# --------- EasyOCR init (single shared reader for speed) ----------
import torch
import easyocr

# --------- PaddleOCR init (single shared reader for speed) ----------
from paddleocr import PaddleOCR, PPStructureV3

LANGS = ["en"]

# Model directory must be specified via environment variable
MODEL_DIR = os.getenv("EASYOCR_MODEL_DIR")
if not MODEL_DIR:
    raise RuntimeError("EASYOCR_MODEL_DIR environment variable must be set")

# GPU selection: must be explicitly set
gpu_env = os.getenv("OCR_USE_GPU")
if gpu_env is None:
    raise RuntimeError("OCR_USE_GPU environment variable must be set (true/false)")
gpu_env = gpu_env.lower()
if gpu_env in ["true", "1", "yes"]:
    USE_GPU = True
elif gpu_env in ["false", "0", "no"]:
    USE_GPU = False
else:
    raise ValueError(f"Invalid OCR_USE_GPU value: {gpu_env}. Must be true/false")

# Check compute capability - fail if below 7.0 (unsupported by this PyTorch build)
if torch.cuda.is_available():
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    if USE_GPU and cc_major < 7:
        raise RuntimeError(f"GPU compute capability {cc_major}.{cc_minor} < 7.0; unsupported by this PyTorch build.")

# Fail fast if GPU requested but unavailable
if USE_GPU and not torch.cuda.is_available():
    raise RuntimeError("OCR_USE_GPU=true but CUDA is not available on this host.")

print(
    f"GPU Detection: CUDA available={torch.cuda.is_available()}, "
    f"OCR_USE_GPU={os.getenv('OCR_USE_GPU', 'auto')}, Using GPU={USE_GPU}"
)
print(f"EasyOCR models will be loaded from: {MODEL_DIR}")

# Verify model directory exists and contains required files
if not os.path.exists(MODEL_DIR):
    raise RuntimeError(f"Model directory does not exist: {MODEL_DIR}")
model_files_exist = all([
    os.path.exists(os.path.join(MODEL_DIR, "craft_mlt_25k.pth")),
    # g2 recognizers cover Latin & Arabic in recent EasyOCR versions
    any(os.path.exists(os.path.join(MODEL_DIR, name)) for name in ("latin_g2.pth", "english_g2.pth")),
    any(os.path.exists(os.path.join(MODEL_DIR, name)) for name in ("arabic_g2.pth", "arabic.pth")),
])

if not model_files_exist:
    raise RuntimeError(f"Required EasyOCR model files not found in {MODEL_DIR}. Please download models manually.")

print("Using existing local models")

READER = easyocr.Reader(
    LANGS,
    gpu=USE_GPU,
    model_storage_directory=MODEL_DIR,
    user_network_directory=MODEL_DIR,
    download_enabled=False,
)
READER_LOCK = Lock()

# --------- PaddleOCR setup ----------
# PaddleOCR model directory (all PaddleOCR models go here)
PADDLEOCR_HOME = os.getenv("PADDLEOCR_HOME")
if not PADDLEOCR_HOME:
    raise RuntimeError("PADDLEOCR_HOME environment variable must be set")

# PaddleOCR GPU selection: must be explicitly set
paddle_gpu_env = os.getenv("PADDLEOCR_USE_GPU")
if paddle_gpu_env is None:
    raise RuntimeError("PADDLEOCR_USE_GPU environment variable must be set (true/false)")
paddle_gpu_env = paddle_gpu_env.lower()
if paddle_gpu_env in ["true", "1", "yes"]:
    PADDLE_USE_GPU = True
elif paddle_gpu_env in ["false", "0", "no"]:
    PADDLE_USE_GPU = False
else:
    raise ValueError(f"Invalid PADDLEOCR_USE_GPU value: {paddle_gpu_env}. Must be true/false")

# Verify PaddleOCR model directory exists
if not os.path.exists(PADDLEOCR_HOME):
    raise RuntimeError(f"PaddleOCR model directory does not exist: {PADDLEOCR_HOME}")

# Note: PADDLE_PDX_MODEL_SOURCE is no longer required as it was unused

# PaddleOCR readers (one per language)
PADDLE_READERS = {}
PADDLE_LOCK = Lock()
STRUCTURE_READER = None
STRUCTURE_LOCK = Lock()

def _get_paddle_reader(lang: str):
    """Get or create PaddleOCR reader for specific language."""
    if not lang:
        raise ValueError("Language parameter is required")
    lang = lang.strip().lower()
    if lang not in PADDLE_READERS:
        with PADDLE_LOCK:
            if lang not in PADDLE_READERS:  # Double-check after acquiring lock
                try:
                    print(f"Initializing PaddleOCR for language: {lang}")
                    PADDLE_READERS[lang] = PaddleOCR(
                        lang=lang,
                        use_textline_orientation=True,
                        enable_mkldnn=not PADDLE_USE_GPU,
                        cpu_threads=4
                    )
                    print(f"PaddleOCR initialized successfully for {lang}")
                except Exception as e:
                    error_msg = f"Failed to initialize PaddleOCR: {str(e)}"
                    print(error_msg)
                    raise RuntimeError(error_msg)
    return PADDLE_READERS[lang]

def to_py(obj):
    """Deep convert NumPy objects to Python types for JSON serialization."""
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        val = obj.item()
        # Handle special float values that aren't JSON compliant
        if isinstance(val, float):
            if math.isnan(val):
                return None
            elif math.isinf(val):
                return None  # Convert both -inf and +inf to None
        return val
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_py(x) for x in obj]
    # Handle regular Python floats that might be inf/nan
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        elif math.isinf(obj):
            return None
    # Paddle result objects expose .res
    if hasattr(obj, "res"):
        return to_py(obj.res)
    return obj

def _get_structure_reader():
    """Get or create PP-StructureV3 reader."""
    global STRUCTURE_READER
    if STRUCTURE_READER is None:
        with STRUCTURE_LOCK:
            if STRUCTURE_READER is None:
                try:
                    print("Initializing PP-StructureV3 for document parsing")
                    STRUCTURE_READER = PPStructureV3(lang="en")
                    print("PP-StructureV3 initialized successfully")
                except Exception as e:
                    error_msg = f"Failed to initialize PP-StructureV3: {str(e)}"
                    print(error_msg)
                    raise RuntimeError(error_msg)
    return STRUCTURE_READER

print(f"PaddleOCR Detection: GPU={PADDLE_USE_GPU}")
print(f"PaddleOCR models will be loaded from: {PADDLEOCR_HOME}")

app = FastAPI(
    title="OCR API with EasyOCR, PaddleOCR + LLaVA Vision",
    version="4.2.0",
    description="""
    Advanced OCR API with multiple processing engines.
    """,
)

# ============================= Vision OCR (Ollama) =============================
# Model: Use LLaVA via Ollama API
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_PORT = os.getenv("OLLAMA_PORT")
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME")

if not OLLAMA_HOST:
    raise RuntimeError("OLLAMA_HOST environment variable must be set")
if not OLLAMA_PORT:
    raise RuntimeError("OLLAMA_PORT environment variable must be set")
if not VISION_MODEL_NAME:
    raise RuntimeError("VISION_MODEL_NAME environment variable must be set")

OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

print(f"Vision model: {VISION_MODEL_NAME}")
print(f"Ollama base URL: {OLLAMA_BASE_URL}")

def check_ollama_connection():
    """Check if Ollama is running and LLaVA model is available."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            if VISION_MODEL_NAME in model_names:
                print(f"Ollama is running and LLaVA model {VISION_MODEL_NAME} is available")
                return True, VISION_MODEL_NAME
            else:
                print(f"LLaVA model {VISION_MODEL_NAME} not found. Available models: {model_names}")
                return False, None
        else:
            print(f"Ollama API returned status {response.status_code}")
            return False, None
    except Exception as e:
        print(f"Ollama connection failed: {e}")
        return False, None

def encode_image_to_base64(image):
    """Convert PIL Image to base64 string for Ollama API."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# Try to connect to Ollama, but don't fail if it's not available
OLLAMA_AVAILABLE, SELECTED_MODEL = check_ollama_connection()
LLAVA_LOCK = Lock() if OLLAMA_AVAILABLE else None

# ------------------------ I/O helpers -----------------------------

def load_image_bgr_from_bytes(data: bytes) -> np.ndarray:
    """Load via PIL to honor EXIF orientation and ICC, then convert to OpenCV BGR."""
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)
        if "icc_profile" in im.info and im.info["icc_profile"]:
            src_profile = ImageCms.ImageCmsProfile(io.BytesIO(im.info["icc_profile"]))
            dst_profile = ImageCms.createProfile("sRGB")
            im = ImageCms.profileToProfile(im, src_profile, dst_profile, outputMode="RGB")
        else:
            im = im.convert("RGB")
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

# ---------- Multilingual routing helpers (EN + AR) ----------
_AR_RANGE = (0x0600, 0x06FF)  # Arabic block

def _has_arabic_chars(s: str) -> bool:
    for ch in s or "":
        cp = ord(ch)
        if _AR_RANGE[0] <= cp <= _AR_RANGE[1]:
            return True
    return False

def _shape_arabic_for_display(text: str) -> str:
    # Correct shaping and RTL ordering for display
    import arabic_reshaper
    from bidi.algorithm import get_display
    return get_display(arabic_reshaper.reshape(text))

def _clip_rect(x, y, w, h, W, H):
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h

def _polys_from_det_result(det_result):
    """
    Accepts PaddleOCR v3-style det-only output and returns a list of polygons.
    Expected keys: 'det_polys'.
    """
    if isinstance(det_result, dict):
        return det_result.get('det_polys', [])
    raise ValueError(f"Unexpected PaddleOCR detection result format: {type(det_result)}")

def _detect_text_polys(bgr: np.ndarray) -> list:
    """
    One detection pass (language-agnostic).
    Reuse any reader (we use 'en') only for detection.
    """
    det_reader = _get_paddle_reader("en")
    det_out = det_reader.ocr(bgr, det=True, rec=False, cls=False)
    if not det_out:
        raise RuntimeError("PaddleOCR detection returned no results")
    first = det_out[0]
    polys = _polys_from_det_result(first)
    # normalize: list[(x,y)...]
    return [[(int(p[0]), int(p[1])) for p in poly] for poly in polys]

def _recognize_crop(reader, crop_bgr: np.ndarray):
    """
    Recognize a small crop. Returns (text, avg_conf).
    """
    out = reader.ocr(crop_bgr, det=False, rec=True, cls=True)
    if not out:
        raise RuntimeError("PaddleOCR recognition returned no results")

    result = out[0]
    # v3 dict format only
    if not isinstance(result, dict) or 'rec_texts' not in result or 'rec_scores' not in result:
        raise ValueError(f"Unexpected PaddleOCR recognition result format: {type(result)}")
    
    texts = result.get('rec_texts') or []
    scores = result.get('rec_scores') or []
    if texts:
        joined = " ".join([t for t in texts if t])
        avg = float(np.mean([float(s) for s in scores])) if scores else 0.0
        return joined, avg
    return "", 0.0

def _route_line_to_lang(en_reader, ar_reader, crop: np.ndarray):
    """
    Recognize same crop with EN and AR; prefer:
    1) script presence (Arabic chars) if similar confidence
    2) otherwise higher confidence
    """
    txt_en, sc_en = _recognize_crop(en_reader, crop) if en_reader else ("", 0.0)
    txt_ar, sc_ar = _recognize_crop(ar_reader, crop) if ar_reader else ("", 0.0)

    # Prefer Arabic if it looks Arabic and isn't clearly worse
    if ar_reader and _has_arabic_chars(txt_ar) and (sc_ar >= sc_en * 0.95):
        return _shape_arabic_for_display(txt_ar), float(sc_ar)

    # Pick higher confidence
    if sc_ar > sc_en:
        txt = _shape_arabic_for_display(txt_ar) if _has_arabic_chars(txt_ar) else txt_ar
        return txt, float(sc_ar)

    return txt_en, float(sc_en)

def _recognize_multilang(bgr: np.ndarray, langs: list[str]) -> list:
    """
    Detect once, then recognize each crop using the best language among langs.
    Returns a list of dicts: {text, confidence, box}
    """
    H, W = bgr.shape[:2]
    polys = _detect_text_polys(bgr)
    if not polys:
        raise RuntimeError("No text polygons detected in image")

    use_en = "en" in langs
    use_ar = "ar" in langs
    en_reader = _get_paddle_reader("en") if use_en else None
    ar_reader = _get_paddle_reader("ar") if use_ar else None

    lines = []
    for poly in polys:
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        x, y = max(0, min(xs)), max(0, min(ys))
        w, h = (max(xs) - x + 1), (max(ys) - y + 1)
        x, y, w, h = _clip_rect(x, y, w, h, W, H)
        crop = bgr[y:y+h, x:x+w].copy()

        if len(langs) == 1:
            reader = en_reader or ar_reader  # whichever exists
            text, conf = _recognize_crop(reader, crop)
            if use_ar and _has_arabic_chars(text):
                text = _shape_arabic_for_display(text)
        else:
            text, conf = _route_line_to_lang(en_reader, ar_reader, crop)

        if text:
            lines.append({
                "text": text,
                "confidence": float(conf),
                "box": [(int(px), int(py)) for px, py in poly],
            })

    # Sort for readability
    lines.sort(key=lambda L: (min(p[1] for p in L["box"]), min(p[0] for p in L["box"])))
    return lines

# ----------------------- Schemas -----------------------------------

class LineOut(BaseModel):
    text: str
    confidence: float
    box: List[Tuple[int, int]]

class OCRResponse(BaseModel):
    full_text: str
    avg_confidence: float
    lines: List[LineOut]

class VisionOCRResponse(BaseModel):
    text: str
    model: str

class PPStructureV3Response(BaseModel):
    """Response model for PP-StructureV3 complex document parsing"""
    res: list

class PPChatOCRv4Response(BaseModel):
    """Response model for PP-ChatOCRv4 intelligent information extraction"""
    answer: str

# ----------------------- Routes ------------------------------------

@app.get("/health", summary="Health Check & System Status", tags=["System"])
def health():
    return {
        "status": "ok",
        "easyocr_gpu": USE_GPU,
        "paddleocr_gpu": PADDLE_USE_GPU,
        "vision_available": OLLAMA_AVAILABLE,
        "vision_model": SELECTED_MODEL,
    }


# ============================= EasyOCR Engine =============================
easyocr_router = APIRouter()

@easyocr_router.post("/", response_model=OCRResponse, summary="Fast OCR with EasyOCR Engine", tags=["EasyOCR"])
async def easyocr_endpoint(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    
    bgr = load_image_bgr_from_bytes(data)
    
    with READER_LOCK:
        results = READER.readtext(bgr, detail=1, paragraph=False)
    
    lines = []
    confs = []
    for (box, text, conf) in results:
        lines.append(LineOut(text=text, confidence=conf, box=[(int(p[0]), int(p[1])) for p in box]))
        confs.append(conf)
        
    full_text = " ".join([line.text for line in lines])
    avg_conf = np.mean(confs) if confs else 0.0
    
    return OCRResponse(full_text=full_text, avg_confidence=avg_conf, lines=lines)

# ============================= LLaVA Engine =============================
llava_router = APIRouter()

@llava_router.post("/", response_model=VisionOCRResponse, summary="Advanced OCR with LLaVA Vision Model", tags=["LLaVA"])
async def llavaocr_endpoint(file: UploadFile = File(...), prompt: str = Query("Extract all text from this image.")):
    if not OLLAMA_AVAILABLE:
        raise HTTPException(status_code=503, detail="LLaVA-OCR via Ollama not available.")
    
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        image_base64 = encode_image_to_base64(im)

    ollama_payload = {
        "model": SELECTED_MODEL,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False,
    }
    
    with LLAVA_LOCK:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=ollama_payload, timeout=180)
    
    if response.status_code == 200:
        result = response.json()
        return VisionOCRResponse(text=result.get("response", ""), model=SELECTED_MODEL)
    else:
        raise HTTPException(status_code=500, detail=f"Ollama API error: {response.text}")

# ============================= PaddleOCR Engine =============================
paddle_router = APIRouter()

@paddle_router.post(
    "/recognize",
    response_model=OCRResponse,
    summary="OCR with per-line language routing (e.g., ?lang=en,ar)",
    tags=["PaddleOCR"],
)
async def paddle_recognize(
    file: UploadFile = File(...),
    lang: str = Query("en", description="Comma-separated list, e.g. 'en' or 'en,ar'"),
):
    """
    One endpoint only:
    - If lang has a single value (e.g., 'en'), we still detect once but only use that recognizer.
    - If lang has multiple values (e.g., 'en,ar'), we detect once and route per line.
    """
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    bgr = load_image_bgr_from_bytes(data)
    langs_list = [s.strip().lower() for s in lang.split(",") if s.strip()]
    if not langs_list:
        raise ValueError("At least one language must be specified")

    # Make sure readers exist (this triggers downloads on first run)
    try:
        for l in set(langs_list):
            _ = _get_paddle_reader(l)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        routed = _recognize_multilang(bgr, langs_list)  # list of dicts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")

    # Convert to your Pydantic model
    lines = [LineOut(text=L["text"], confidence=L["confidence"], box=L["box"]) for L in routed]
    confs = [L.confidence for L in lines]
    full_text = " ".join([L.text for L in lines])
    avg_conf = float(np.mean(confs)) if confs else 0.0

    return OCRResponse(full_text=full_text, avg_confidence=avg_conf, lines=lines)

@paddle_router.post(
    "/structure",
    summary="Complex Document Parsing (PP-StructureV3) + bilingual OCR overlay",
    tags=["PaddleOCR"],
)
async def paddle_structure(
    file: UploadFile = File(...),
    lang: str = Query("en", description="Comma-separated languages for recognition overlay, e.g. 'en' or 'en,ar'"),
):
    """
    Runs PP-StructureV3 to parse layout/tables, then runs one multilingual OCR pass
    (detect-once, per-line routed recognition) over the whole page using the languages
    listed in ?lang=...
    Returns JSON with both 'structure' and 'ocr' keys.
    """
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    bgr = load_image_bgr_from_bytes(data)

    # Parse and validate languages (trigger model downloads up front)
    langs_list = [s.strip().lower() for s in lang.split(",") if s.strip()]
    if not langs_list:
        raise ValueError("At least one language must be specified")
    try:
        for l in set(langs_list):
            _ = _get_paddle_reader(l)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Run PP-StructureV3
    try:
        struct_reader = _get_structure_reader()
        struct_out = struct_reader.predict(input=bgr)   # list[Result]
        struct_serializable = [to_py(o) for o in struct_out]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Structure analysis failed: {str(e)}")

    # Run bilingual OCR overlay (detect once, route EN/AR per line)
    try:
        routed_lines = _recognize_multilang(bgr, langs_list)  # list of dicts {text, confidence, box}
        # Build summary
        confs = [float(L["confidence"]) for L in routed_lines]
        full_text = " ".join([L["text"] for L in routed_lines])
        avg_conf = float(np.mean(confs)) if confs else 0.0
        ocr_overlay = {
            "langs": langs_list,
            "full_text": full_text,
            "avg_confidence": avg_conf,
            "lines": routed_lines,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR overlay failed: {str(e)}")

    # Return JSON (bypass Pydantic so numpy never leaks)
    return JSONResponse(content=jsonable_encoder({
        "structure": struct_serializable,
        "ocr": ocr_overlay,
    }))

@paddle_router.post("/chat", response_model=PPChatOCRv4Response, summary="Intelligent Information Extraction with PP-ChatOCRv4", tags=["PaddleOCR"])
async def paddle_chat(file: UploadFile = File(...), question: str = Query("Extract all key information.")):
    # This is a simplified implementation. A real implementation would involve a VQA model.
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    
    bgr = load_image_bgr_from_bytes(data)
    
    reader = _get_paddle_reader('en')

    results = reader.ocr(bgr)
    
    full_text = ""
    if results and len(results) > 0:
        result = results[0]
        if 'rec_texts' in result:
            full_text = " ".join(result['rec_texts'])

    # Simple logic to find keywords from the question in the text
    answer = f"Could not find an answer to '{question}'."
    if full_text:
        # A more sophisticated implementation would use an LLM here.
        # For now, we just return the full text as a proxy for an answer.
        answer = f"Based on the document, here is the extracted text that might answer your question '{question}':\n\n{full_text}"

    return PPChatOCRv4Response(answer=answer)

# Register the routers
app.include_router(easyocr_router, prefix="/easyocr")
app.include_router(llava_router, prefix="/llavaocr")
app.include_router(paddle_router, prefix="/paddleocr")

@app.on_event("startup")
def warmup_paddle():
    """
    Warmup PaddleOCR models on startup for production deployment.
    This pre-downloads models and avoids slow first requests.
    """
    print("Starting PaddleOCR warmup...")
    # Warmup PaddleOCR with English model
    r = _get_paddle_reader("en")
    dummy = np.full((32, 32, 3), 255, dtype=np.uint8)
    _ = r.ocr(dummy)
    print("✓ PaddleOCR English model warmed up")

    # Warmup PP-Structure
    s = _get_structure_reader()
    _ = s.predict(input=dummy)
    print("✓ PP-StructureV3 warmed up")
    
    print("✅ PaddleOCR warmup complete.")

# Optional: run with `python main.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
