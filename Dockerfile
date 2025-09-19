# === Base image ===
FROM python:3.12-bookworm

# ---------- Build args (override at build time) ----------
# Choose OpenCV flavor: "opencv-python-headless" (default) or "opencv-python"
ARG OPENCV_PACKAGE=opencv-python-headless
# Optional: PyTorch CUDA wheels index (e.g., https://download.pytorch.org/whl/cu121)
# Leave empty for CPU-only PyTorch from PyPI.
ARG TORCH_EXTRA_INDEX_URL=

# ---------- System setup ----------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OS libs needed for OpenCV (and general runtime)
# - libgl1 + libglib2.0-0 for OpenCV GUI variant
# - libsm6, libxext6, libxrender1 are common extras some OpenCV ops expect
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling early
RUN python -m pip install -U pip setuptools wheel

# ---------- Python deps ----------
# We write an explicit requirements file (without OpenCV, Torch variants handled below)
# Your requested packages:
# annotated-types, anyio, certifi, click, dnspython, easyocr, email-validator,
# fastapi, fastapi-cli, fastapi-cloud-cli, filelock, fsspec, h11, httpcore,
# httptools, httpx, idna, imageio, Jinja2, lazy_loader, markdown-it-py, MarkupSafe,
# mdurl, mpmath, networkx, ninja, numpy, packaging, pillow, pip, pyclipper,
# pydantic, pydantic_core, Pygments, python-bidi, python-dotenv, python-multipart,
# PyYAML, rich, rich-toolkit, rignore, scikit-image, scipy, sentry-sdk, setuptools,
# shapely, shellingham, sniffio, starlette, sympy, tifffile, typer,
# typing_extensions, typing-inspection, urllib3, uvicorn, uvloop, watchfiles, websockets
# Plus the NVIDIA CUDA runtime/aux libs (cu12 namespace) you requested.
RUN bash -lc 'cat > /tmp/requirements.txt << "REQS"\n\
annotated-types\n\
anyio\n\
certifi\n\
click\n\
dnspython\n\
easyocr\n\
email-validator\n\
fastapi\n\
fastapi-cli\n\
fastapi-cloud-cli\n\
filelock\n\
fsspec\n\
h11\n\
httpcore\n\
httptools\n\
httpx\n\
idna\n\
imageio\n\
Jinja2\n\
lazy_loader\n\
markdown-it-py\n\
MarkupSafe\n\
mdurl\n\
mpmath\n\
networkx\n\
ninja\n\
numpy\n\
nvidia-cublas-cu12\n\
nvidia-cuda-cupti-cu12\n\
nvidia-cuda-nvrtc-cu12\n\
nvidia-cuda-runtime-cu12\n\
nvidia-cudnn-cu12\n\
nvidia-cufft-cu12\n\
nvidia-cufile-cu12\n\
nvidia-curand-cu12\n\
nvidia-cusolver-cu12\n\
nvidia-cusparse-cu12\n\
nvidia-cusparselt-cu12\n\
nvidia-nccl-cu12\n\
nvidia-nvjitlink-cu12\n\
nvidia-nvtx-cu12\n\
packaging\n\
pillow\n\
pip\n\
pyclipper\n\
pydantic\n\
pydantic_core\n\
Pygments\n\
python-bidi\n\
python-dotenv\n\
python-multipart\n\
PyYAML\n\
rich\n\
rich-toolkit\n\
rignore\n\
scikit-image\n\
scipy\n\
sentry-sdk\n\
setuptools\n\
shapely\n\
shellingham\n\
sniffio\n\
starlette\n\
sympy\n\
tifffile\n\
typer\n\
typing_extensions\n\
typing-inspection\n\
urllib3\n\
uvicorn\n\
uvloop\n\
watchfiles\n\
websockets\n\
REQS'

# Install base requirements
RUN python -m pip install -r /tmp/requirements.txt

# Ensure uvicorn + watchfiles are installed for FastAPI/uvicorn reload
RUN python -m pip install --no-cache-dir uvicorn watchfiles


# Install OpenCV flavor (headless by default, or GUI if you pass OPENCV_PACKAGE=opencv-python)
RUN python - <<PY
import os, sys, subprocess
pkg = os.environ.get("OPENCV_PACKAGE", "opencv-python-headless")
print(f"Installing {pkg} ...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
PY

# Install PyTorch (+ torchvision + triton)
# - CPU default from PyPI
# - If TORCH_EXTRA_INDEX_URL is provided (e.g., cu121), use it to get CUDA wheels.
RUN bash -lc 'if [ -n "$TORCH_EXTRA_INDEX_URL" ]; then \
      python -m pip install --extra-index-url "$TORCH_EXTRA_INDEX_URL" torch torchvision triton ; \
    else \
      python -m pip install torch torchvision triton ; \
    fi'

# ---------- Runtime user & app skeleton ----------
WORKDIR /app
# (Optional) create a non-root user for security
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

# Copy your app (adjust as needed)
# COPY . /app

# Expose FastAPI default port
EXPOSE 8000

# Default command (adjust to your module/app entrypoint)
# Example assumes `main.py` with `app = FastAPI()`
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
