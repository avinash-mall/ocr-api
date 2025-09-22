#!/usr/bin/env python3
"""
Preload all OCR models (PaddleOCR + EasyOCR) to avoid runtime downloads.
This script downloads all required models during Docker build.
"""

import os
import sys
import easyocr
from paddleocr import PaddleOCR, PPStructureV3

def main():
    """Preload all required OCR models."""
    print("Starting OCR model preloading...")
    
    # Set environment variables - use Docker paths if available, otherwise local paths
    if os.path.exists("/opt"):
        # Running in Docker
        paddleocr_home = "/opt/paddleocr_models"
        easyocr_home = "/opt/easyocr_models"
    else:
        # Running locally
        paddleocr_home = os.path.abspath("paddleocr_models")
        easyocr_home = os.path.abspath("easyocr_models")
    
    os.environ["PADDLEOCR_HOME"] = paddleocr_home
    os.environ["EASYOCR_MODEL_DIR"] = easyocr_home
    
    # Create the directories if they don't exist
    os.makedirs(paddleocr_home, exist_ok=True)
    os.makedirs(easyocr_home, exist_ok=True)
    print(f"Using PaddleOCR home: {paddleocr_home}")
    print(f"Using EasyOCR home: {easyocr_home}")
    
    try:
        # Preload EasyOCR models
        print("\n=== Preloading EasyOCR models ===")
        print("Preloading EasyOCR English model...")
        easyocr_reader = easyocr.Reader(['en'], model_storage_directory=easyocr_home, download_enabled=True)
        print("✓ EasyOCR English model loaded")
        
        print("Preloading EasyOCR Arabic model...")
        easyocr_reader_ar = easyocr.Reader(['ar'], model_storage_directory=easyocr_home, download_enabled=True)
        print("✓ EasyOCR Arabic model loaded")
        
        # Preload PaddleOCR models
        print("\n=== Preloading PaddleOCR models ===")
        print("Preloading English model...")
        en_reader = PaddleOCR(lang="en")
        print("✓ PaddleOCR English model loaded")
        
        print("Preloading Arabic model...")
        ar_reader = PaddleOCR(lang="ar")
        print("✓ PaddleOCR Arabic model loaded")
        
        print("Preloading PP-StructureV3...")
        structure_reader = PPStructureV3(lang="en")
        print("✓ PP-StructureV3 loaded")
        
        print("\n✅ All OCR models preloaded successfully!")
        print(f"PaddleOCR models: {paddleocr_home}")
        print(f"EasyOCR models: {easyocr_home}")
        
    except Exception as e:
        print(f"❌ Error preloading models: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
