# Development script for OCR API - uses existing images with volume mounts

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("cpu", "gpu", "stop", "logs", "restart")]
    [string]$Action = "gpu"
)

switch ($Action) {
    "cpu" {
        Write-Host "Starting CPU container with code mounted..." -ForegroundColor Green
        docker stop ocr-dev-cpu 2>$null
        docker rm ocr-dev-cpu 2>$null
        docker run -d --name ocr-dev-cpu `
            -p 8000:8000 `
            -v "${PWD}/main.py:/app/main.py:ro" `
            -v "${PWD}/easyocr_models:/app/easyocr_models" `
            -e OCR_USE_GPU=false `
            ocr-api:cpu
        Write-Host "CPU container running on http://localhost:8000" -ForegroundColor Yellow
    }
    "gpu" {
        Write-Host "Starting GPU container with code mounted..." -ForegroundColor Green
        docker stop ocr-dev-gpu 2>$null
        docker rm ocr-dev-gpu 2>$null
        docker run -d --name ocr-dev-gpu `
            --gpus all `
            -p 8001:8000 `
            -v "${PWD}/main.py:/app/main.py:ro" `
            -v "${PWD}/easyocr_models:/app/easyocr_models" `
            -e OCR_USE_GPU=true `
            ocr-api:gpu
        Write-Host "GPU container running on http://localhost:8001" -ForegroundColor Yellow
    }
    "stop" {
        Write-Host "Stopping development containers..." -ForegroundColor Red
        docker stop ocr-dev-cpu ocr-dev-gpu 2>$null
        docker rm ocr-dev-cpu ocr-dev-gpu 2>$null
        Write-Host "Development containers stopped" -ForegroundColor Green
    }
    "logs" {
        Write-Host "Showing logs for GPU development container..." -ForegroundColor Cyan
        docker logs -f ocr-dev-gpu
    }
    "restart" {
        Write-Host "Restarting GPU container to pick up code changes..." -ForegroundColor Blue
        docker restart ocr-dev-gpu
        Write-Host "Container restarted! Wait a few seconds for startup." -ForegroundColor Yellow
    }
}

if ($Action -ne "stop" -and $Action -ne "logs" -and $Action -ne "restart") {
    Write-Host "`nTo check health:" -ForegroundColor Cyan
    if ($Action -eq "cpu") {
        Write-Host "curl http://localhost:8000/health" -ForegroundColor White
    } else {
        Write-Host "curl http://localhost:8001/health" -ForegroundColor White
    }
    Write-Host "`nTo view logs:" -ForegroundColor Cyan
    Write-Host "./dev.ps1 logs" -ForegroundColor White
    Write-Host "`nTo restart after code changes:" -ForegroundColor Cyan
    Write-Host "./dev.ps1 restart" -ForegroundColor White
    Write-Host "`nNo rebuild needed - just restart to pick up code changes!" -ForegroundColor Green
}
