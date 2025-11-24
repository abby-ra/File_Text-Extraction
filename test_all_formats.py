"""
Test script to demonstrate all supported document formats
"""
from pathlib import Path
import sys

def test_format_support():
    """Display all supported formats and test with available files"""
    
    print("=" * 80)
    print("DOCUMENT FORMAT SUPPORT TEST")
    print("=" * 80)
    
    # All supported formats
    formats = {
        "Images": [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".gif"],
        "PDF Documents": [".pdf"],
        "Microsoft Word": [".docx", ".doc"],
        "Microsoft PowerPoint": [".pptx", ".ppt"],
        "Microsoft Excel": [".xlsx", ".xls"],
        "Plain Text": [".txt"],
        "Spreadsheets": [".csv"],
        "Rich Text": [".rtf"],
        "Web Documents": [".html", ".htm"],
        "OpenDocument": [".odt"]
    }
    
    print("\n📋 SUPPORTED FORMATS:")
    print("-" * 80)
    for category, extensions in formats.items():
        ext_list = ", ".join(extensions)
        print(f"  {category:20s} : {ext_list}")
    
    print("\n" + "=" * 80)
    print("SCANNING INPUT FILES")
    print("=" * 80)
    
    input_dir = Path("input_files")
    if not input_dir.exists():
        print("❌ input_files directory not found!")
        return
    
    # Get all files
    all_files = list(input_dir.iterdir())
    
    # Categorize files
    categorized = {}
    unsupported = []
    
    for file in all_files:
        if file.is_file():
            ext = file.suffix.lower()
            
            # Find category
            found = False
            for category, extensions in formats.items():
                if ext in extensions:
                    if category not in categorized:
                        categorized[category] = []
                    categorized[category].append(file.name)
                    found = True
                    break
            
            if not found and ext:
                unsupported.append(file.name)
    
    # Display results
    print(f"\n📁 Total files in input_files/: {len(all_files)}")
    print("-" * 80)
    
    for category in formats.keys():
        if category in categorized:
            files = categorized[category]
            print(f"\n✅ {category} ({len(files)} files):")
            for fname in files[:5]:  # Show first 5
                print(f"   - {fname}")
            if len(files) > 5:
                print(f"   ... and {len(files) - 5} more")
    
    if unsupported:
        print(f"\n⚠️  Unsupported formats ({len(unsupported)} files):")
        for fname in unsupported[:5]:
            print(f"   - {fname}")
        if len(unsupported) > 5:
            print(f"   ... and {len(unsupported) - 5} more")
    
    print("\n" + "=" * 80)
    print("EXTRACTION CAPABILITIES")
    print("=" * 80)
    
    capabilities = [
        ("📄 Text Documents", "Direct text extraction from DOCX, TXT, PDF (text layer)"),
        ("🖼️  Images & Scans", "OCR using PaddleOCR, EasyOCR, TrOCR, Tesseract"),
        ("📊 Spreadsheets", "Cell-by-cell extraction from Excel, CSV files"),
        ("🎨 Presentations", "Slide-by-slide text extraction from PowerPoint"),
        ("🌐 Web Pages", "HTML parsing with BeautifulSoup"),
        ("📝 Rich Text", "RTF parsing with pypandoc or regex fallback"),
        ("🔍 Scanned PDFs", "Automatic OCR for image-based PDF pages"),
        ("🖼️  Image Description", "Qwen-VL vision language model analysis"),
        ("🔄 Structured Output", "JSON export with Docling converter"),
        ("🌍 Malayalam Support", "Multi-engine OCR with Unicode preservation")
    ]
    
    for icon_title, description in capabilities:
        print(f"\n{icon_title}")
        print(f"  └─ {description}")
    
    print("\n" + "=" * 80)
    print("USAGE")
    print("=" * 80)
    print("""
To extract text from all supported formats:

    python enhanced_extract.py

Features:
  • Processes all files in input_files/ directory
  • Saves extracted text to output/ directory
  • Creates {filename}.txt for each input file
  • Generates {filename}_description.txt for images (Qwen-VL)
  • Creates {filename}.json for structured data (Docling)
  • Handles multiple formats automatically
  • Preserves Unicode characters (Malayalam, etc.)

Output Structure:
  output/
    ├── document.txt          (extracted text)
    ├── document.json         (structured data)
    ├── image.txt             (OCR text)
    └── image_description.txt (AI description)
""")
    
    print("=" * 80)

if __name__ == "__main__":
    test_format_support()
