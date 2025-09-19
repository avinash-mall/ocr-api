# PaddleOCR Model Download Troubleshooting

## Issue: Network Connectivity Problems

If you encounter errors like:
- `NameResolutionError: Failed to resolve 'paddleocr.bj.bcebos.com'`
- `ConnectionError: HTTPSConnectionPool(host='paddleocr.bj.bcebos.com', port=443)`
- `SSL: UNEXPECTED_EOF_WHILE_READING`

## Root Causes

1. **Network Restrictions**: Corporate firewalls or network policies blocking the domain
2. **DNS Issues**: DNS resolution problems in containerized environments
3. **SSL/TLS Issues**: Certificate validation problems
4. **Geographic Restrictions**: Some regions may have limited access

## Official Model URLs (Still Active)

The official PaddleOCR model server is still operational:
- **Base URL**: `https://paddleocr.bj.bcebos.com/`
- **PP-OCRv3 English Detection**: `https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar`
- **PP-OCRv3 English Recognition**: `https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar`

## Alternative Solutions

### 1. Manual Model Download

Download models manually and place them in the correct directory structure:

```bash
# Create directory structure
mkdir -p paddleocr_models/det
mkdir -p paddleocr_models/rec
mkdir -p paddleocr_models/cls

# Download models manually (if you have access)
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar -O paddleocr_models/det/en_PP-OCRv3_det_infer.tar
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar -O paddleocr_models/rec/en_PP-OCRv3_rec_infer.tar

# Extract models
cd paddleocr_models/det && tar -xf en_PP-OCRv3_det_infer.tar
cd ../rec && tar -xf en_PP-OCRv3_rec_infer.tar
```

### 2. Alternative Download Sources

#### Hugging Face
Some PaddleOCR models are available on Hugging Face:
- Search for "PaddleOCR" on https://huggingface.co/
- Look for models like "PaddlePaddle/PP-OCRv5_server_det"

#### GitHub Releases
Check the official PaddleOCR GitHub repository for releases:
- https://github.com/PaddlePaddle/PaddleOCR/releases
- Models may be available as release assets

#### Google Drive Mirrors
Some users report success with Google Drive mirrors mentioned in GitHub issues.

### 3. Docker Network Configuration

If running in Docker, ensure proper network configuration:

```yaml
# docker-compose.yaml
services:
  ocr-api-cpu:
    # ... other config
    dns:
      - 8.8.8.8
      - 8.8.4.4
    extra_hosts:
      - "paddleocr.bj.bcebos.com:YOUR_IP_ADDRESS"
```

### 4. Proxy Configuration

If behind a corporate proxy, configure PaddleOCR to use it:

```python
import os
os.environ['HTTP_PROXY'] = 'http://your-proxy:port'
os.environ['HTTPS_PROXY'] = 'http://your-proxy:port'
```

### 5. Pre-download Models in Dockerfile

Add model download to your Dockerfile:

```dockerfile
# Add to Dockerfile before running the application
RUN mkdir -p /app/paddleocr_models/det /app/paddleocr_models/rec /app/paddleocr_models/cls
RUN wget https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar -O /app/paddleocr_models/det/en_PP-OCRv3_det_infer.tar || echo "Download failed, will retry at runtime"
RUN wget https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar -O /app/paddleocr_models/rec/en_PP-OCRv3_rec_infer.tar || echo "Download failed, will retry at runtime"
```

## Current Error Handling

The application now includes:

1. **Network Connectivity Check**: Tests general internet access
2. **PaddleOCR Server Check**: Specifically tests access to paddleocr.bj.bcebos.com
3. **Detailed Error Messages**: Clear explanation of the issue
4. **Alternative Source Suggestions**: Mentions Hugging Face and GitHub as alternatives
5. **Graceful Degradation**: Creates dummy readers that fail with clear error messages

## Testing Connectivity

You can test connectivity from within the container:

```bash
# Test general internet connectivity
docker exec ocr-api-cpu curl -I https://www.google.com

# Test PaddleOCR server specifically
docker exec ocr-api-cpu curl -I https://paddleocr.bj.bcebos.com/

# Test DNS resolution
docker exec ocr-api-cpu nslookup paddleocr.bj.bcebos.com
```

## Recommended Actions

1. **Check Network**: Ensure the container has internet access
2. **Test DNS**: Verify DNS resolution works
3. **Check Firewall**: Ensure no firewall rules block the domain
4. **Use Alternatives**: Consider Hugging Face or manual download
5. **Pre-download**: Download models during Docker build if possible

## Model Directory Structure

When manually downloading models, ensure this structure:

```
paddleocr_models/
├── det/
│   └── en_PP-OCRv3_det_infer/
│       ├── inference.pdiparams
│       ├── inference.pdiparams.info
│       └── inference.pdmodel
├── rec/
│   └── en_PP-OCRv3_rec_infer/
│       ├── inference.pdiparams
│       ├── inference.pdiparams.info
│       └── inference.pdmodel
└── cls/
    └── (classification models if needed)
```
