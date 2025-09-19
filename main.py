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
import time
import base64
import requests
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image, ImageOps, ImageCms
from threading import Lock
import tempfile
import pathlib

# --------- EasyOCR init (single shared reader for speed) ----------
import torch
import easyocr

# --------- PaddleOCR init (single shared reader for speed) ----------
try:
    from paddleocr import PaddleOCR
    _HAS_PADDLE = True
    _PADDLE_IMPORT_ERROR = None
except ImportError as e:
    _HAS_PADDLE = False
    _PADDLE_IMPORT_ERROR = str(e)
    PaddleOCR = None

LANGS = ["en", "ar"]

# Allow overriding the model dir via env; default to ./easyocr_models
MODEL_DIR = os.getenv(
    "EASYOCR_MODEL_DIR",
    os.path.join(os.path.dirname(__file__), "easyocr_models"),
)

# GPU selection: env first, then CUDA availability, then capability (>= sm_70)
gpu_env = os.getenv("OCR_USE_GPU", "auto").lower()
if gpu_env in ["true", "1", "yes"]:
    USE_GPU = True
elif gpu_env in ["false", "0", "no"]:
    USE_GPU = False
else:
    USE_GPU = torch.cuda.is_available()

# Check compute capability and force CPU if below 7.0 (unsupported by this PyTorch build)
if torch.cuda.is_available():
    try:
        cc_major, cc_minor = torch.cuda.get_device_capability(0)
        if USE_GPU and cc_major < 7:
            print(f"GPU compute capability {cc_major}.{cc_minor} < 7.0; forcing CPU for compatibility.")
            USE_GPU = False
    except Exception as _cap_err:
        # If capability check fails, assume unsupported to be safe
        if USE_GPU:
            print(f"GPU capability check failed ({_cap_err}); forcing CPU for safety.")
            USE_GPU = False

print(
    f"GPU Detection: CUDA available={torch.cuda.is_available()}, "
    f"OCR_USE_GPU={os.getenv('OCR_USE_GPU', 'auto')}, Using GPU={USE_GPU}"
)
print(f"EasyOCR models will be loaded from: {MODEL_DIR}")

# Ensure local model directory exists and detect presence of core weights
os.makedirs(MODEL_DIR, exist_ok=True)
model_files_exist = all([
    os.path.exists(os.path.join(MODEL_DIR, "craft_mlt_25k.pth")),
    # g2 recognizers cover Latin & Arabic in recent EasyOCR versions
    any(os.path.exists(os.path.join(MODEL_DIR, name)) for name in ("latin_g2.pth", "english_g2.pth")),
    any(os.path.exists(os.path.join(MODEL_DIR, name)) for name in ("arabic_g2.pth", "arabic.pth")),
])

download_enabled = not model_files_exist
print("Using existing local models" if not download_enabled else "Models not found locally, downloading...")

READER = easyocr.Reader(
    LANGS,
    gpu=USE_GPU,
    model_storage_directory=MODEL_DIR,
    user_network_directory=MODEL_DIR,
    download_enabled=download_enabled,
)
READER_LOCK = Lock()

# --------- PaddleOCR setup ----------
PADDLE_MODEL_DIR = os.getenv(
    "PADDLEOCR_MODEL_DIR",
    os.path.join(os.path.dirname(__file__), "paddleocr_models"),
)

# PaddleOCR GPU selection
paddle_gpu_env = os.getenv("PADDLEOCR_USE_GPU", "auto").lower()
if paddle_gpu_env in ["true", "1", "yes"]:
    PADDLE_USE_GPU = True
elif paddle_gpu_env in ["false", "0", "no"]:
    PADDLE_USE_GPU = False
else:
    PADDLE_USE_GPU = torch.cuda.is_available()

# Ensure PaddleOCR model directory exists
os.makedirs(PADDLE_MODEL_DIR, exist_ok=True)

# PaddleOCR readers (one per language)
PADDLE_READERS = {}
PADDLE_LOCK = Lock()

def _get_paddle_reader(lang: str):
    """Get or create PaddleOCR reader for specific language."""
    if not _HAS_PADDLE:
        raise RuntimeError(f"PaddleOCR not available: {_PADDLE_IMPORT_ERROR}")
    
    if lang not in PADDLE_READERS:
        with PADDLE_LOCK:
            if lang not in PADDLE_READERS:  # Double-check after acquiring lock
                PADDLE_READERS[lang] = PaddleOCR(
                    use_angle_cls=True,
                    use_gpu=PADDLE_USE_GPU,
                    lang=lang,
                    show_log=False,
                    det_model_dir=os.path.join(PADDLE_MODEL_DIR, "det"),
                    rec_model_dir=os.path.join(PADDLE_MODEL_DIR, "rec"),
                    cls_model_dir=os.path.join(PADDLE_MODEL_DIR, "cls"),
                )
    return PADDLE_READERS[lang]

print(f"PaddleOCR Detection: Available={_HAS_PADDLE}, GPU={PADDLE_USE_GPU if _HAS_PADDLE else 'N/A'}")
if _HAS_PADDLE:
    print(f"PaddleOCR models will be loaded from: {PADDLE_MODEL_DIR}")

app = FastAPI(
    title="OCR API with EasyOCR + LLaVA Vision",
    version="4.0.0",
    description="""
    Advanced OCR API with dual processing engines:
    
    **EasyOCR Engine (/easyocr):**
    - Fast, lightweight OCR for English/Arabic text
    - Optimized for forms, certificates, and documents
    - CPU/GPU support with preprocessing variants
    
    **LLaVA Vision Engine (/llavaocr):**
    - Advanced vision-language model via Ollama
    - Superior text extraction with context understanding
    - Supports multiple languages including Arabic
    - Handles complex layouts and mixed content
    
    Choose the right engine for your use case!
    """,
)

# ============================= Vision OCR (Ollama) =============================
# Model: Use LLaVA or qariocr via Ollama API
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")  # Default to localhost for local development
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "llava:latest")
FALLBACK_MODEL_NAME = os.getenv("FALLBACK_MODEL_NAME", "qariocr:latest")

print(f"Primary vision model: {VISION_MODEL_NAME}")
print(f"Fallback vision model: {FALLBACK_MODEL_NAME}")
print(f"Ollama base URL: {OLLAMA_BASE_URL}")

def check_ollama_connection():
    """Check if Ollama is running and vision models are available."""
    try:
        # Check if Ollama is running
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            
            primary_available = VISION_MODEL_NAME in model_names
            fallback_available = FALLBACK_MODEL_NAME in model_names
            
            if primary_available:
                print(f"Ollama is running and primary model {VISION_MODEL_NAME} is available")
                return True, VISION_MODEL_NAME
            elif fallback_available:
                print(f"Ollama is running and fallback model {FALLBACK_MODEL_NAME} is available")
                return True, FALLBACK_MODEL_NAME
            else:
                print(f"Ollama is running but no vision models found. Available models: {model_names}")
                return False, None
        else:
            print(f"Ollama API returned status {response.status_code}")
            return False, None
    except requests.exceptions.RequestException as e:
        print(f"Cannot connect to Ollama at {OLLAMA_BASE_URL}: {e}")
        return False, None

def encode_image_to_base64(image):
    """Convert PIL Image to base64 string for Ollama API."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')

# Check Ollama connection and get available model
OLLAMA_AVAILABLE, SELECTED_MODEL = check_ollama_connection()
QARI_LOCK = Lock()

# ------------------------ I/O helpers -----------------------------

def load_image_bgr_from_bytes(data: bytes) -> np.ndarray:
    """Load via PIL to honor EXIF orientation and ICC, then convert to OpenCV BGR."""
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)  # respect EXIF orientation
        try:
            if "icc_profile" in im.info and im.info["icc_profile"]:
                src_profile = ImageCms.ImageCmsProfile(bytes(im.info.get("icc_profile")))
                dst_profile = ImageCms.createProfile("sRGB")
                im = ImageCms.profileToProfile(im, src_profile, dst_profile, outputMode="RGB")
            else:
                im = im.convert("RGB")
        except Exception:
            im = im.convert("RGB")
        arr = np.array(im)  # RGB
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def resize_to_multiple(img: np.ndarray, max_side: int = 1920, multiple: int = 32, allow_upscale: bool = True) -> np.ndarray:
    """Resize keeping aspect ratio; final H/W are multiples of `multiple`."""
    h, w = img.shape[:2]
    base_scale = max_side / max(h, w)
    scale = base_scale if (allow_upscale or base_scale < 1.0) else 1.0
    nh, nw = int(h * scale), int(w * scale)
    nh = max(multiple, (nh // multiple) * multiple or multiple)
    nw = max(multiple, (nw // multiple) * multiple or multiple)
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(img, (nw, nh), interpolation=interp)

def unsharp_mask(img: np.ndarray, sigma: float = 1.0, amount: float = 0.5) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)

def gamma_correct(img: np.ndarray, gamma: float = 1.12) -> np.ndarray:
    inv = 1.0 / max(gamma, 1e-6)
    table = (np.linspace(0, 1, 256) ** inv) * 255.0
    return cv2.LUT(img, table.astype(np.uint8))

def illumination_correct(gray: np.ndarray, kernel: int = 31) -> np.ndarray:
    bg = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    return cv2.normalize(cv2.subtract(gray, bg), None, 0, 255, cv2.NORM_MINMAX)

def deskew_by_hough(gray: np.ndarray, max_angle_deg: float = 15.0):
    edges = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 1800, threshold=200,
                            minLineLength=max(gray.shape) // 8, maxLineGap=20)
    angle_deg = 0.0
    if lines is not None and len(lines) > 0:
        angles = []
        for (x1, y1, x2, y2) in lines[:, 0]:
            dx, dy = x2 - x1, y2 - y1
            if dx == 0:
                continue
            ang = math.degrees(math.atan2(dy, dx))
            if -max_angle_deg <= ang <= max_angle_deg:
                angles.append(ang)
        if angles:
            angle_deg = float(np.median(angles))
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# ----------------- Preprocess for EasyOCR -------------------------

def preprocess_color_for_easyocr(bgr: np.ndarray, max_side: int, allow_upscale: bool) -> np.ndarray:
    """Color path helps detector; keep structure and color cues."""
    # Gray-world white balance
    b, g, r = [np.mean(bgr[:, :, c]) for c in range(3)]
    gray = (b + g + r) / 3.0
    scales = (gray / (b + 1e-6), gray / (g + 1e-6), gray / (r + 1e-6))
    wb = bgr.astype(np.float32).copy()
    for c, s in enumerate(scales):
        wb[:, :, c] *= s
    wb = np.clip(wb, 0, 255).astype(np.uint8)

    wb = gamma_correct(wb, gamma=1.12)
    wb = cv2.fastNlMeansDenoisingColored(wb, None, h=5, hColor=5, templateWindowSize=7, searchWindowSize=21)

    lab = cv2.cvtColor(wb, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(L)
    enh = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

    enh = unsharp_mask(enh, sigma=1.0, amount=0.5)
    enh = resize_to_multiple(enh, max_side=max_side, multiple=32, allow_upscale=allow_upscale)
    return enh

def preprocess_gray_for_easyocr(bgr: np.ndarray, max_side: int, allow_upscale: bool, deskew: bool = True) -> np.ndarray:
    """Gray variant for low-contrast docs; avoid hard binarization to keep edges."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    gray = illumination_correct(gray, kernel=31)
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    if deskew:
        gray = deskew_by_hough(gray)
    gray = unsharp_mask(gray, sigma=1.0, amount=0.5)
    gray = resize_to_multiple(gray, max_side=max_side, multiple=32, allow_upscale=allow_upscale)
    return gray

def preprocess_binarized_for_easyocr(bgr: np.ndarray, max_side: int, allow_upscale: bool) -> np.ndarray:
    """Adaptive binarization for messy backgrounds / shaded scans."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = illumination_correct(gray, kernel=31)
    # Adaptive threshold (Gaussian) – block size odd, C tunes background subtraction
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, blockSize=25, C=15)
    # Clean specks: small opening then slight closing
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    th = resize_to_multiple(th, max_side=max_side, multiple=32, allow_upscale=allow_upscale)
    return th

# ----------------------- OCR runner --------------------------------

def run_easyocr(
    reader,
    img: np.ndarray,
    *,
    paragraph: bool,
    decoder: str,
    beam_width: int,
    min_size: int,
    text_threshold: float,
    low_text: float,
    link_threshold: float,
    contrast_ths: float,
    adjust_contrast: float,
    allowlist: Optional[str],
    blocklist: Optional[str],
):
    """
    EasyOCR accepts numpy arrays (OpenCV BGR or grayscale).
    Returns boxes, texts, confidences, avg_conf.
    """
    results = reader.readtext(
        img,
        detail=1,
        paragraph=paragraph,
        rotation_info=(0, 90, 180, 270),
        decoder=decoder,
        beamWidth=beam_width,
        min_size=min_size,
        text_threshold=text_threshold,
        low_text=low_text,
        link_threshold=link_threshold,
        contrast_ths=contrast_ths,
        adjust_contrast=adjust_contrast,
        allowlist=allowlist,
        blocklist=blocklist,
    )
    boxes, texts, confs = [], [], []
    for item in results:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            box, txt, conf = item[0], item[1], float(item[2])
            box = [(int(x), int(y)) for x, y in box]
            boxes.append(box)
            texts.append(txt)
            confs.append(conf)
    avg_conf = float(np.mean(confs)) if confs else 0.0
    return boxes, texts, confs, avg_conf

def _ocr_on_variants(
    reader,
    images: dict,
    *,
    paragraph: bool,
    decoder: str,
    beam_width: int,
    min_size: int,
    text_threshold: float,
    low_text: float,
    link_threshold: float,
    contrast_ths: float,
    adjust_contrast: float,
    allowlist: Optional[str],
    blocklist: Optional[str],
):
    """Run OCR on multiple preprocessed variants and return the best by avg_conf."""
    best = ("", [], [], [], 0.0)
    for name, im in images.items():
        with READER_LOCK:
            boxes, texts, confs, avg = run_easyocr(
                reader, im,
                paragraph=paragraph,
                decoder=decoder,
                beam_width=beam_width,
                min_size=min_size,
                text_threshold=text_threshold,
                low_text=low_text,
                link_threshold=link_threshold,
                contrast_ths=contrast_ths,
                adjust_contrast=adjust_contrast,
                allowlist=allowlist,
                blocklist=blocklist,
            )
        if avg >= best[4]:
            best = (name, boxes, texts, confs, avg)
    return best

# ----------------------- PaddleOCR helpers -------------------------

def _run_paddleocr(reader, img: np.ndarray):
    """
    Run PaddleOCR on a single image.
    Returns boxes, texts, confidences, avg_conf.
    """
    results = reader.ocr(img, cls=True)
    boxes, texts, confs = [], [], []
    
    if results and results[0]:
        for line in results[0]:
            if line and len(line) >= 2:
                box, (text, conf) = line[0], line[1]
                # Convert box format to match EasyOCR format
                box = [(int(x), int(y)) for x, y in box]
                boxes.append(box)
                texts.append(text)
                confs.append(float(conf))
    
    avg_conf = float(np.mean(confs)) if confs else 0.0
    return boxes, texts, confs, avg_conf

def _paddle_on_variants(
    reader,
    images: dict,
):
    """Run PaddleOCR on multiple preprocessed variants and return the best by avg_conf."""
    best = ("", [], [], [], 0.0)
    for name, im in images.items():
        with PADDLE_LOCK:
            boxes, texts, confs, avg = _run_paddleocr(reader, im)
        if avg >= best[4]:
            best = (name, boxes, texts, confs, avg)
    return best

# ----------------------- Schemas -----------------------------------

class LineOut(BaseModel):
    text: str
    confidence: float
    box: List[Tuple[int, int]]

class OCRResponse(BaseModel):
    full_text: str
    avg_confidence: float
    variant_used: str
    lines: List[LineOut]

class VisionOCRResponse(BaseModel):
    """Response model for vision-based OCR endpoints (LLaVA, etc.)"""
    text: str
    model: str
    max_new_tokens: int
    prompt_used: str
    cache_dir: str
    device: str

# Legacy alias for backward compatibility
QariOCRResponse = VisionOCRResponse

# ----------------------- Routes ------------------------------------

@app.get("/health",
    summary="Health Check & System Status",
    description="""
    **System health and configuration endpoint**
    
    Returns current status of both OCR engines and system configuration.
    
    **Response includes:**
    - Overall system status
    - EasyOCR engine status (GPU/CPU mode)
    - LLaVA vision engine availability
    - Model directories and configurations
    - Ollama connection status
    """,
    tags=["System"]
)
def health():
    return {
        "status": "ok",
        "gpu": USE_GPU,
        "langs": LANGS,
        "model_dir": MODEL_DIR,
        "model_dir_exists": os.path.isdir(MODEL_DIR),
        "vision_model": SELECTED_MODEL if OLLAMA_AVAILABLE else "None",
        "ollama_url": OLLAMA_BASE_URL,
        "vision_available": OLLAMA_AVAILABLE,
        "paddle_available": _HAS_PADDLE,
        "paddle_gpu": PADDLE_USE_GPU if _HAS_PADDLE else False,
        "paddle_model_dir": PADDLE_MODEL_DIR if _HAS_PADDLE else None,
    }

@app.post("/easyocr", response_model=OCRResponse,
    summary="Fast OCR with EasyOCR Engine",
    description="""
    **EasyOCR-powered endpoint** for fast and efficient text extraction.
    
    Optimized for speed and accuracy on forms, certificates, and structured documents.
    Uses multiple preprocessing variants for maximum reliability.
    
    **Features:**
    - Fast processing with CPU/GPU support
    - English and Arabic language support
    - Multiple preprocessing strategies (COLOR, GRAY, BINARIZED)
    - Confidence scoring and line-by-line extraction
    - Optimized for forms and certificates
    
    **Best for:**
    - High-volume processing
    - Structured documents (forms, IDs, certificates)
    - When speed is more important than advanced understanding
    - Clear, well-formatted text
    
    **Processing Strategy:**
    1. COLOR: White-balanced, denoised, CLAHE enhanced
    2. GRAY: Illumination-corrected, deskewed (fallback if confidence < min_conf)
    3. BINARIZED: Adaptive threshold (final fallback)
    """,
    tags=["EasyOCR", "Fast OCR"]
)
async def ocr_endpoint(
    file: UploadFile = File(..., description="JPEG/PNG image with English/Arabic text"),

    # --- Accuracy-first defaults ---
    min_conf: float = Query(0.80, ge=0.0, le=1.0, description="If below this after pass #1, try second-pass upscale."),
    max_side: int = Query(1920, ge=256, le=4096, description="Max image side before snapping to multiple of 32."),
    allow_upscale: bool = Query(True, description="Allow upscaling small images (improves tiny text)."),
    paragraph: bool = Query(False, description="Group lines into paragraphs (keep False for forms)."),
    try_all_variants: bool = Query(True, description="Try color+gray+binarized and pick best; set False for speed."),

    # EasyOCR knobs (reasonable accuracy defaults)
    decoder: str = Query("beamsearch", pattern="^(greedy|beamsearch|wordbeamsearch)$"),
    beam_width: int = Query(7, ge=1, le=20),
    min_size: int = Query(10, ge=5, le=100),
    text_threshold: float = Query(0.7, ge=0.1, le=0.9),
    low_text: float = Query(0.4, ge=0.1, le=0.9),
    link_threshold: float = Query(0.4, ge=0.1, le=0.9),
    contrast_ths: float = Query(0.10, ge=0.0, le=0.5, description="Per-region low-contrast retry threshold."),
    adjust_contrast: float = Query(0.50, ge=0.0, le=1.0, description="Per-region contrast adjustment power."),

    # Optional field constraints to boost accuracy in known contexts
    allowlist: Optional[str] = Query(None, description="Restrict characters (e.g., 0-9 for numeric fields)."),
    blocklist: Optional[str] = Query(None, description="Exclude characters (rarely needed)."),
):
    data = await file.read()
    if not data:
        return JSONResponse(status_code=400, content={"detail": "Empty file."})

    # Load & preprocess (three variants)
    bgr = load_image_bgr_from_bytes(data)
    color_img = preprocess_color_for_easyocr(bgr, max_side=max_side, allow_upscale=allow_upscale)
    gray_img  = preprocess_gray_for_easyocr(bgr, max_side=max_side, allow_upscale=allow_upscale, deskew=True)
    bin_img   = preprocess_binarized_for_easyocr(bgr, max_side=max_side, allow_upscale=allow_upscale)

    if try_all_variants:
        variant_used, boxes, texts, confs, avg_conf = _ocr_on_variants(
            READER,
            {"color": color_img, "gray": gray_img, "binarized": bin_img},
            paragraph=paragraph,
            decoder=decoder,
            beam_width=beam_width,
            min_size=min_size,
            text_threshold=text_threshold,
            low_text=low_text,
            link_threshold=link_threshold,
            contrast_ths=contrast_ths,
            adjust_contrast=adjust_contrast,
            allowlist=allowlist,
            blocklist=blocklist,
        )
    else:
        # Fast path: color only
        with READER_LOCK:
            boxes, texts, confs, avg_conf = run_easyocr(
                READER, color_img,
                paragraph=paragraph,
                decoder=decoder,
                beam_width=beam_width,
                min_size=min_size,
                text_threshold=text_threshold,
                low_text=low_text,
                link_threshold=link_threshold,
                contrast_ths=contrast_ths,
                adjust_contrast=adjust_contrast,
                allowlist=allowlist,
                blocklist=blocklist,
            )
        variant_used = "color"

    # Second-pass upscale if still weak and upscaling allowed
    if avg_conf < min_conf and allow_upscale and max_side < 2560:
        boosted = min(2560, max(2048, int(max_side * 1.33)))
        color2 = preprocess_color_for_easyocr(bgr, max_side=boosted, allow_upscale=True)
        gray2  = preprocess_gray_for_easyocr(bgr, max_side=boosted, allow_upscale=True, deskew=True)
        bin2   = preprocess_binarized_for_easyocr(bgr, max_side=boosted, allow_upscale=True)
        if try_all_variants:
            v2, b2, t2, c2, a2 = _ocr_on_variants(
                READER, {"color": color2, "gray": gray2, "binarized": bin2},
                paragraph=paragraph,
                decoder=decoder,
                beam_width=beam_width,
                min_size=min_size,
                text_threshold=text_threshold,
                low_text=low_text,
                link_threshold=link_threshold,
                contrast_ths=contrast_ths,
                adjust_contrast=adjust_contrast,
                allowlist=allowlist,
                blocklist=blocklist,
            )
        else:
            with READER_LOCK:
                b2, t2, c2, a2 = run_easyocr(
                    READER, color2,
                    paragraph=paragraph,
                    decoder=decoder,
                    beam_width=beam_width,
                    min_size=min_size,
                    text_threshold=text_threshold,
                    low_text=low_text,
                    link_threshold=link_threshold,
                    contrast_ths=contrast_ths,
                    adjust_contrast=adjust_contrast,
                    allowlist=allowlist,
                    blocklist=blocklist,
                )
            v2 = "color"
        if a2 > avg_conf:
            variant_used, boxes, texts, confs, avg_conf = v2, b2, t2, c2, a2

    lines = [LineOut(text=t, confidence=float(c), box=b) for b, t, c in zip(boxes, texts, confs)]
    full_text = "\n".join(texts)

    return OCRResponse(
        full_text=full_text,
        avg_confidence=float(avg_conf),
        variant_used=variant_used,
        lines=lines
    )

# ----------------------------- /paddleocr -----------------------------------
@app.post("/paddleocr", response_model=OCRResponse,
    summary="Fast OCR with PaddleOCR Engine",
    description="""
    **PaddleOCR-powered endpoint** providing an alternative OCR engine to EasyOCR.

    Mirrors the core behavior of `/easyocr`:
    - English and Arabic language support (select via `lang` query)
    - CPU/GPU support and auto-download of models into a local cache
    - Multiple preprocessing variants (COLOR, GRAY, BINARIZED) with best-variant selection
    - Confidence scoring and line-by-line extraction compatible with your existing schema
    """,
    tags=["PaddleOCR", "Fast OCR"]
)
async def paddleocr_endpoint(
    file: UploadFile = File(..., description="JPEG/PNG image with text"),
    lang: str = Query("en", description="Language for PaddleOCR model (e.g., 'en' or 'ar')."),
    min_conf: float = Query(0.80, ge=0.0, le=1.0, description="If below this after pass #1, try second-pass upscale."),
    max_side: int = Query(1920, ge=256, le=4096, description="Max image side before snapping to multiple of 32."),
    allow_upscale: bool = Query(True, description="Allow upscaling small images (improves tiny text)."),
    try_all_variants: bool = Query(True, description="Try color+gray+binarized and pick best; set False for speed."),
):
    data = await file.read()
    if not data:
        return JSONResponse(status_code=400, content={"detail": "Empty file."})

    if not _HAS_PADDLE:
        return JSONResponse(status_code=503, content={"detail": f"PaddleOCR not available: {_PADDLE_IMPORT_ERROR}"})

    # Load & preprocess (three variants) — reuse the same helpers as EasyOCR
    bgr = load_image_bgr_from_bytes(data)
    color_img = preprocess_color_for_easyocr(bgr, max_side=max_side, allow_upscale=allow_upscale)
    gray_img  = preprocess_gray_for_easyocr(bgr, max_side=max_side, allow_upscale=allow_upscale, deskew=True)
    bin_img   = preprocess_binarized_for_easyocr(bgr, max_side=max_side, allow_upscale=allow_upscale)

    reader = _get_paddle_reader(lang)

    if try_all_variants:
        variant_used, boxes, texts, confs, avg_conf = _paddle_on_variants(
            reader, {"color": color_img, "gray": gray_img, "binarized": bin_img}
        )
    else:
        with PADDLE_LOCK:
            boxes, texts, confs, avg_conf = _run_paddleocr(reader, color_img)
        variant_used = "color"

    # Second-pass upscale if still weak and upscaling allowed (parity with /easyocr)
    if avg_conf < min_conf and allow_upscale and max_side < 2560:
        boosted = min(2560, max(2048, int(max_side * 1.33)))
        color2 = preprocess_color_for_easyocr(bgr, max_side=boosted, allow_upscale=True)
        gray2  = preprocess_gray_for_easyocr(bgr, max_side=boosted, allow_upscale=True, deskew=True)
        bin2   = preprocess_binarized_for_easyocr(bgr, max_side=boosted, allow_upscale=True)
        if try_all_variants:
            v2, b2, t2, c2, a2 = _paddle_on_variants(reader, {"color": color2, "gray": gray2, "binarized": bin2})
        else:
            with PADDLE_LOCK:
                b2, t2, c2, a2 = _run_paddleocr(reader, color2)
            v2 = "color"
        if a2 > avg_conf:
            variant_used, boxes, texts, confs, avg_conf = v2, b2, t2, c2, a2

    lines = [LineOut(text=t, confidence=float(c), box=b) for b, t, c in zip(boxes, texts, confs)]
    full_text = "\n".join(texts)

    return OCRResponse(
        full_text=full_text,
        avg_confidence=float(avg_conf),
        variant_used=variant_used,
        lines=lines,
    )

# ----------------------------- /llavaocr -----------------------------------
@app.post("/llavaocr", response_model=QariOCRResponse,
    summary="Advanced OCR with LLaVA Vision Model",
    description="""
    **LLaVA-powered OCR endpoint** for superior text extraction from images.
    
    This endpoint uses the LLaVA (Large Language and Vision Assistant) model via Ollama 
    to provide advanced OCR capabilities with context understanding.
    
    **Features:**
    - Multi-language support (Arabic, English, and more)
    - Context-aware text extraction
    - Handles complex layouts and mixed content
    - Superior accuracy for challenging documents
    
    **Best for:**
    - Complex documents with mixed languages
    - Images with challenging layouts
    - When you need high accuracy over speed
    - Documents requiring context understanding
    
    **Requirements:**
    - Ollama running with LLaVA model installed
    - Image format: JPEG, PNG
    - Max file size: recommended < 10MB
    """,
    tags=["LLaVA OCR", "Vision AI"]
)
async def llavaocr_endpoint(
    file: UploadFile = File(..., description="Image file (JPEG/PNG) containing text to extract. Supports multiple languages including Arabic and English."),
    prompt: str = Query(
        "Extract all text from this image.",
        description="Custom prompt for the vision model. Use 'Extract all text' for OCR, or customize for specific extraction needs."
    ),
    max_new_tokens: int = Query(
        2000, 
        ge=16, 
        le=4096, 
        description="Maximum number of tokens to generate. Higher values allow longer text extraction."
    ),
    do_sample: bool = Query(
        False, 
        description="Enable sampling for more creative responses. Keep False for deterministic OCR results."
    ),
    temperature: float = Query(
        0.2, 
        ge=0.0, 
        le=2.0, 
        description="Sampling temperature. Lower values (0.0-0.2) for precise OCR, higher for creative text."
    ),
    top_p: float = Query(
        0.95, 
        ge=0.0, 
        le=1.0, 
        description="Top-p sampling parameter. Only used when do_sample=True."
    ),
):
    print(f"[LLAVA-OCR] ===== Starting request =====")
    print(f"[LLAVA-OCR] File: {file.filename}, size: {file.size}")
    print(f"[LLAVA-OCR] Prompt: {prompt[:100]}...")
    print(f"[LLAVA-OCR] Generation params - max_tokens: {max_new_tokens}, do_sample: {do_sample}, temp: {temperature}, top_p: {top_p}")
    
    if not OLLAMA_AVAILABLE:
        print("[LLAVA-OCR] ERROR: Ollama not available")
        return JSONResponse(status_code=503, content={"detail": "LLaVA-OCR via Ollama not available. Check if Ollama is running and vision models are installed."})

    print("[LLAVA-OCR] Reading file data...")
    data = await file.read()
    print(f"[LLAVA-OCR] File data size: {len(data)} bytes")
    
    if not data:
        print("[LLAVA-OCR] ERROR: Empty file")
        return JSONResponse(status_code=400, content={"detail": "Empty file."})

    # Load PIL image (respect EXIF like elsewhere)
    print("[LLAVA-OCR] Loading and processing image...")
    with Image.open(io.BytesIO(data)) as im:
        print(f"[LLAVA-OCR] Original image mode: {im.mode}, size: {im.size}")
        im = ImageOps.exif_transpose(im).convert("RGB")
        pil_image = im.copy()
        print(f"[LLAVA-OCR] Processed image mode: {pil_image.mode}, size: {pil_image.size}")

    # Encode image to base64 for Ollama API
    print("[LLAVA-OCR] Encoding image to base64...")
    image_base64 = encode_image_to_base64(pil_image)
    print(f"[LLAVA-OCR] Image encoded, base64 length: {len(image_base64)}")

    # Prepare Ollama API request
    print(f"[LLAVA-OCR] Preparing Ollama API request for model: {SELECTED_MODEL}...")
    ollama_payload = {
        "model": SELECTED_MODEL,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False,
        "options": {
            "num_predict": max_new_tokens,
            "temperature": temperature if do_sample else 0.0,
            "top_p": top_p if do_sample else 1.0,
        }
    }
    print(f"[LLAVA-OCR] Ollama payload prepared with {len(ollama_payload['images'])} image(s)")

    print("[LLAVA-OCR] Sending request to Ollama...")
    start_time = time.time()
    try:
        with QARI_LOCK:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=ollama_payload,
                timeout=180  # 3 minute timeout for large models
            )
        
        generation_time = time.time() - start_time
        print(f"[LLAVA-OCR] Ollama response received in {generation_time:.2f}s")
        print(f"[LLAVA-OCR] Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            out_text = result.get("response", "")
            print(f"[LLAVA-OCR] Response length: {len(out_text)}")
            print(f"[LLAVA-OCR] Response preview: {out_text[:200]}...")
            print(f"[LLAVA-OCR] ===== Request completed =====")
            
            return QariOCRResponse(
                text=out_text,
                model=SELECTED_MODEL,
                max_new_tokens=max_new_tokens,
                prompt_used=prompt,
                cache_dir="ollama",
                device="ollama",
            )
        else:
            print(f"[LLAVA-OCR] ERROR: Ollama API returned status {response.status_code}")
            print(f"[LLAVA-OCR] Response: {response.text}")
            return JSONResponse(status_code=500, content={"detail": f"Ollama API error: {response.status_code}"})
            
    except requests.exceptions.RequestException as e:
        print(f"[LLAVA-OCR] ERROR: Request to Ollama failed: {e}")
        return JSONResponse(status_code=500, content={"detail": f"Failed to connect to Ollama: {str(e)}"})
    except Exception as e:
        print(f"[LLAVA-OCR] ERROR: Unexpected error: {e}")
        import traceback
        print(f"[LLAVA-OCR] Traceback: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"detail": f"Unexpected error: {str(e)}"})

# Optional: run with `python main.py` (instead of uvicorn CLI)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
