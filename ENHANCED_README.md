# Enhanced Malayalam Text Extraction Pipeline

## Overview
This enhanced extraction pipeline supports **Malayalam** text extraction using **Chithrakan** for both printed and handwritten Malayalam content, with Tesseract as a fallback for mixed-script documents.

## Features

### Multi-Language Support
- ✅ **Malayalam**: Chithrakan (primary engine for printed & handwritten)
- ✅ **English**: PaddleOCR
- ✅ **Handwritten**: TrOCR (microsoft/trocr-large-handwritten)
- ✅ **Mixed Content**: Tesseract fallback (eng+mal)

### Supported Formats
- **Images**: PNG, JPG, JPEG, BMP, TIFF, WEBP
- **Documents**: PDF (text + scanned), DOCX, DOC, PPTX, XLSX, TXT
- **Special**: Job-cards, scanned photos, handwritten notes

### Output Formats
- **Text files**: UTF-8 encoded `.txt` preserving original script
- **Structured data**: JSON format via Docling integration
- **Multi-engine results**: Shows which OCR engine extracted the text

## Installation

### 1. Install Python Dependencies
```bash
pip install -r requirements_enhanced.txt
```

### 2. Install Tesseract OCR (for fallback)
**Windows:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Install and add to PATH
- Download Malayalam language data: `mal.traineddata`

**Linux:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-mal
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

### 3. Install Chithrakan
```bash
pip install chithrakan
```

If Chithrakan is not available via pip, clone and install from source:
```bash
git clone https://github.com/AI4Bharat/Chithrakan.git
cd Chithrakan
pip install -e .
```

### 4. Install Docling (Optional - for structured output)
```bash
pip install docling
```

## Usage

### Basic Usage
```bash
python enhanced_extract.py
```

This will:
1. Process all files in `input_files/` directory
2. Extract text using appropriate engines
3. Save results to `output/` directory
4. Generate structured JSON files (if Docling available)

### File Organization
```
sih-doc/
├── input_files/           # Place your files here
│   ├── malayalam_doc.jpg
│   ├── handwritten.png
│   ├── mixed_content.pdf
│   └── ...
├── output/                # Extracted text appears here
│   ├── malayalam_doc.txt
│   ├── malayalam_doc.json (structured)
│   └── ...
└── enhanced_extract.py
```

## OCR Engine Priority

### For Images
1. **Chithrakan** (Malayalam - printed & handwritten)
2. **PaddleOCR** (English)
3. **TrOCR** (Handwritten content)
4. **Tesseract** (Mixed content fallback)

### For PDFs
1. Direct text extraction (if text-based)
2. Page-by-page OCR for scanned pages
3. All OCR engines applied as needed

### For DOCX
1. python-docx (direct extraction)
2. docx2txt (fallback)
3. Image extraction + OCR (for embedded images)

## Output Format

### Text Files (.txt)
```
[Malayalam - Chithrakan]
മലയാളം ടെക്സ്റ്റ് ഇവിടെ...

[English - PaddleOCR]
English text here...

[Handwritten - TrOCR]
Handwritten text here...
```

### JSON Files (.json)
```json
{
  "source": "document.txt",
  "text": "Full extracted text...",
  "structured": {
    "sections": [...],
    "metadata": {...}
  }
}
```

## Troubleshooting

### Malayalam text not extracting
- Ensure Chithrakan is properly installed
- Check if image quality is sufficient
- Try Tesseract fallback: verify `mal.traineddata` is installed

### Handwriting not recognized
- TrOCR model is large (~2.2GB) - ensure sufficient memory
- Works best on clear handwriting
- May need fine-tuning for specific handwriting styles

### PDF extraction fails
- Check if PDF is corrupted: `python -c "import fitz; fitz.open('file.pdf')"`
- For password-protected PDFs, decrypt first
- Some PDFs may need repair tools

### DOCX extraction fails
- File may be corrupted or use old DOC format
- Try saving as new DOCX from Word/LibreOffice
- Check if file is actually a ZIP file: `unzip -t file.docx`

## Performance Tips

1. **Batch Processing**: Place all files in `input_files/` for batch processing
2. **Image Quality**: Higher resolution = better OCR (300 DPI recommended)
3. **Memory**: TrOCR and transformers need ~4-6GB RAM
4. **GPU**: Enable CUDA for faster processing (PyTorch with GPU support)

## API Integration

To use specific engines programmatically:

```python
from enhanced_extract import extract_text_from_image

# Extract from image file
text = extract_text_from_image("path/to/image.png")
print(text)
```

## Known Limitations

1. **Chithrakan availability**: May require installation from source
2. **Malayalam PaddleOCR**: Not available, using Chithrakan instead
3. **Corrupt files**: sample.pdf and some DOCX files may be unrecoverable
4. **Large models**: TrOCR requires significant download (~2.2GB)

## Comparison: Simple vs Enhanced

| Feature | simple_extract.py | enhanced_extract.py |
|---------|------------------|---------------------|
| Malayalam OCR | ❌ (PaddleOCR unavailable) | ✅ Chithrakan |
| Handwriting | ⚠️ (TrOCR loaded, not integrated) | ✅ Fully integrated |
| Mixed scripts | ❌ | ✅ Tesseract fallback |
| Structured output | ❌ | ✅ Docling JSON |
| Multi-engine | ❌ | ✅ 4 engines |
| Script preservation | ⚠️ | ✅ UTF-8 Malayalam preserved |

## Next Steps

1. Test on Malayalam documents
2. Verify Chithrakan installation
3. Install Tesseract for fallback support
4. Test handwritten Malayalam notes
5. Review structured JSON outputs

## Support

For issues with:
- **Chithrakan**: https://github.com/AI4Bharat/Chithrakan
- **PaddleOCR**: https://github.com/PaddlePaddle/PaddleOCR
- **Tesseract**: https://github.com/tesseract-ocr/tesseract
- **Docling**: https://github.com/DS4SD/docling
