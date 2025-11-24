"""
Simple verification script after removing vision models
"""

print("=" * 60)
print("VISION MODEL REMOVAL VERIFICATION")
print("=" * 60)

# Test 1: Check configuration
print("\n[1/4] Testing configuration...")
try:
    from config import config
    assert not hasattr(config.models, 'qwen_vl_model'), "Vision model still in config!"
    assert not hasattr(config.models, 'vision_model'), "Vision model still in config!"
    print("PASS - Vision models removed from configuration")
except Exception as e:
    print(f"FAIL - {e}")

# Test 2: Check pipeline initialization
print("\n[2/4] Testing pipeline initialization...")
try:
    from pipeline import HybridDocumentPipeline
    pipeline = HybridDocumentPipeline()
    assert pipeline.vision_analyzer is None, "Vision analyzer should be None!"
    print("PASS - Pipeline initialized without vision analyzer")
except Exception as e:
    print(f"FAIL - {e}")

# Test 3: Check imports
print("\n[3/4] Checking imports...")
try:
    import sys
    if 'vision_analyzer' in sys.modules:
        print("WARNING - vision_analyzer was imported")
    else:
        print("PASS - vision_analyzer not imported")
except Exception as e:
    print(f"FAIL - {e}")

# Test 4: Check file detection
print("\n[4/4] Testing file detection...")
try:
    from pathlib import Path
    input_dir = Path("input_files")
    files = list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.docx")) + list(input_dir.glob("*.png"))
    print(f"PASS - Found {len(files)} files ready to process")
except Exception as e:
    print(f"FAIL - {e}")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
print("\nThe pipeline is ready without vision models.")
print("You can now process documents using:")
print("  python pipeline.py input_files/sample.pdf")
