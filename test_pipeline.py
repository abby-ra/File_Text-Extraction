"""
Quick test script for the hybrid pipeline
Tests basic configuration and module imports
"""
import sys
from pathlib import Path

def test_imports():
    """Test if all core modules can be imported"""
    print("Testing module imports...")
    
    try:
        import config
        print("✅ config.py imported successfully")
    except Exception as e:
        print(f"❌ config.py import failed: {e}")
        return False
    
    try:
        import preprocessor
        print("✅ preprocessor.py imported successfully")
    except Exception as e:
        print(f"❌ preprocessor.py import failed: {e}")
    
    try:
        import layout_analyzer
        print("✅ layout_analyzer.py imported successfully")
    except Exception as e:
        print(f"❌ layout_analyzer.py import failed: {e}")
    
    try:
        import ocr_engine
        print("✅ ocr_engine.py imported successfully")
    except Exception as e:
        print(f"❌ ocr_engine.py import failed: {e}")
    
    try:
        import office_processor
        print("✅ office_processor.py imported successfully")
    except Exception as e:
        print(f"❌ office_processor.py import failed: {e}")
    
    try:
        import normalizer
        print("✅ normalizer.py imported successfully")
    except Exception as e:
        print(f"❌ normalizer.py import failed: {e}")
    
    try:
        import vision_analyzer
        print("✅ vision_analyzer.py imported successfully")
    except Exception as e:
        print(f"❌ vision_analyzer.py import failed: {e}")
    
    try:
        import reasoning_engine
        print("✅ reasoning_engine.py imported successfully")
    except Exception as e:
        print(f"❌ reasoning_engine.py import failed: {e}")
    
    try:
        import pipeline
        print("✅ pipeline.py imported successfully")
    except Exception as e:
        print(f"❌ pipeline.py import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from config import config
        print(f"✅ Configuration loaded")
        print(f"   - OCR languages: {config.ocr.languages}")
        print(f"   - Preprocessing enabled: {config.processing.enable_preprocessing}")
        print(f"   - Output directory: {config.output.output_dir}")
        print(f"   - Output format: {config.output.output_format}")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_detection():
    """Test file detection in input_files directory"""
    print("\nTesting file detection...")
    
    input_dir = Path("input_files")
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return False
    
    file_patterns = ["*.pdf", "*.docx", "*.doc", "*.xlsx", "*.xls", 
                    "*.pptx", "*.ppt", "*.png", "*.jpg", "*.jpeg"]
    
    files = []
    for pattern in file_patterns:
        files.extend(input_dir.glob(pattern))
    
    if files:
        print(f"✅ Found {len(files)} processable files:")
        for f in files[:5]:  # Show first 5
            print(f"   - {f.name}")
        if len(files) > 5:
            print(f"   ... and {len(files) - 5} more")
    else:
        print("⚠️  No processable files found in input_files/")
    
    return True

def test_pipeline_initialization():
    """Test pipeline initialization"""
    print("\nTesting pipeline initialization...")
    
    try:
        from pipeline import HybridDocumentPipeline
        from config import config
        
        # Initialize with minimal config (disable advanced features for testing)
        config.processing.enable_preprocessing = False
        config.processing.enable_vision_analysis = False
        config.processing.enable_reasoning = False
        
        pipeline = HybridDocumentPipeline(config)
        print("✅ Pipeline initialized successfully")
        print(f"   - Preprocessor: {'enabled' if config.processing.enable_preprocessing else 'disabled'}")
        print(f"   - Vision analysis: {'enabled' if config.processing.enable_vision_analysis else 'disabled'}")
        print(f"   - Reasoning: {'enabled' if config.processing.enable_reasoning else 'disabled'}")
        return True
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("HYBRID DOCUMENT PROCESSING PIPELINE - TEST SUITE")
    print("=" * 60)
    
    results = []
    
    results.append(("Module Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("File Detection", test_file_detection()))
    results.append(("Pipeline Initialization", test_pipeline_initialization()))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Copy .env.example to .env and configure API keys")
        print("2. Install optional dependencies (PaddleOCR, transformers, etc.)")
        print("3. Run: python pipeline.py input_files/ --output output")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
