# Environment Setup for Local Development

## Required Environment Variables

The application requires the following environment variables to be set:

### EasyOCR Configuration
```bash
export EASYOCR_MODEL_DIR=./easyocr_models
export OCR_USE_GPU=false  # or true for GPU
```

### PaddleOCR Configuration
```bash
export PADDLEOCR_HOME=./paddleocr_models
export PADDLEOCR_USE_GPU=false  # or true for GPU
export PADDLE_PDX_MODEL_SOURCE=HF  # or BOS
```


### Ollama Configuration
```bash
export OLLAMA_HOST=localhost
export OLLAMA_PORT=11434
export VISION_MODEL_NAME=llava
```

## Model Directories

The following directories have been created and are ready for model storage:

- `./paddleocr_models/` - All PaddleOCR models (EN/AR, V5, StructureV3, ChatOCRv4)
- `./easyocr_models/` - EasyOCR models (already populated)

**Note:** LLaVA uses external Ollama API calls, so no local model directory is needed.

## Docker vs Local Development

- **Docker**: Models are pre-baked into the image at build time
- **Local**: Models will be downloaded to the local directories on first use
