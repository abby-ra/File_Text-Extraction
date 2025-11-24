"""
Comparison utility to demonstrate enhanced extraction vs simple extraction
"""
import os
from pathlib import Path

def compare_outputs():
    """Compare outputs from both extraction methods"""
    
    print("="*70)
    print("EXTRACTION PIPELINE COMPARISON")
    print("="*70)
    print()
    
    # Check output directory
    output_dir = Path("output")
    
    if not output_dir.exists():
        print("[ERROR] Output directory not found. Run extraction first.")
        return
    
    # Count files
    txt_files = list(output_dir.glob("*.txt"))
    json_files = list(output_dir.glob("*.json"))
    
    print(f"Output Statistics:")
    print(f"  Text files (.txt): {len(txt_files)}")
    print(f"  JSON files (.json): {len(json_files)}")
    print()
    
    # Feature comparison
    print("Feature Comparison:")
    print("-" * 70)
    
    features = [
        ("Malayalam OCR", "❌ Not available", "✅ Chithrakan ready"),
        ("Handwriting OCR", "⚠️  Loaded only", "✅ Fully integrated"),
        ("Multi-engine", "❌ Single engine", "✅ 4 OCR engines"),
        ("Structured output", "❌ Text only", "✅ JSON + Text"),
        ("Script preservation", "⚠️  Basic", "✅ UTF-8 Malayalam"),
        ("Scanned PDFs", "⚠️  Basic OCR", "✅ Page-by-page OCR"),
        ("Error handling", "⚠️  Basic", "✅ Graceful fallbacks"),
        ("Engine labels", "❌ No", "✅ Shows which engine used"),
    ]
    
    print(f"{'Feature':<25} {'simple_extract.py':<25} {'enhanced_extract.py':<25}")
    print("-" * 70)
    for feature, simple, enhanced in features:
        print(f"{feature:<25} {simple:<25} {enhanced:<25}")
    
    print()
    print("="*70)
    
    # Show sample outputs
    if txt_files:
        print()
        print("Sample Output Comparison:")
        print("-" * 70)
        
        sample_file = txt_files[0]
        print(f"File: {sample_file.name}")
        print()
        
        # Read first few lines
        with open(sample_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:10]
            for line in lines:
                print(f"  {line.rstrip()}")
        
        print()
        
        # Check if JSON exists
        json_file = output_dir / f"{sample_file.stem}.json"
        if json_file.exists():
            print(f"✅ Structured JSON available: {json_file.name}")
            print(f"   Size: {json_file.stat().st_size:,} bytes")
        else:
            print("❌ No structured JSON (simple extraction)")
    
    print()
    print("="*70)
    
    # OCR Engine Usage Summary
    print()
    print("OCR Engine Usage:")
    print("-" * 70)
    
    engines = {
        "Chithrakan": 0,
        "PaddleOCR": 0,
        "TrOCR": 0,
        "Tesseract": 0
    }
    
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if "Chithrakan" in content:
                engines["Chithrakan"] += 1
            if "PaddleOCR" in content:
                engines["PaddleOCR"] += 1
            if "TrOCR" in content:
                engines["TrOCR"] += 1
            if "Tesseract" in content:
                engines["Tesseract"] += 1
    
    for engine, count in engines.items():
        if count > 0:
            print(f"  {engine}: {count} files")
        else:
            print(f"  {engine}: Not used")
    
    print()
    print("="*70)

if __name__ == "__main__":
    compare_outputs()
