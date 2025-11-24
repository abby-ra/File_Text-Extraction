from paddleocr import PaddleOCR

# Test Indic languages
indic_langs = {
    'ta': 'Tamil',
    'te': 'Telugu', 
    'ka': 'Kannada',
    'hindi': 'Hindi',
    'devanagari': 'Devanagari',
    'bengali': 'Bengali',
    'malayalam': 'Malayalam',
    'ml': 'Malayalam (ml code)'
}

print("Testing PaddleOCR Indic language support:")
print("-" * 50)

for code, name in indic_langs.items():
    try:
        ocr = PaddleOCR(lang=code, show_log=False)
        print(f"✓ {name} ({code}): SUPPORTED")
    except ValueError as e:
        print(f"✗ {name} ({code}): NOT AVAILABLE")
    except Exception as e:
        print(f"✗ {name} ({code}): ERROR - {str(e)[:50]}")
