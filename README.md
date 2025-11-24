# Hybrid Document Processing Pipeline

A comprehensive document processing system that extracts text and insights from all types of documents including:
- 📄 PDFs (printed and scanned)
- 📝 Office documents (DOCX, XLSX, PPTX)
- 📸 Images (JPG, PNG, TIFF)
- 📐 Engineering drawings
- ✍️ Handwritten notes
- 🌏 Malayalam and English text

## Architecture

The pipeline uses a **hybrid approach** combining multiple specialized engines:

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT DOCUMENTS                          │
│  PDF | DOCX | XLSX | PPTX | JPG | PNG | Handwritten | etc  │
└─────────────────────┬───────────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │  MinerU Preprocessing   │  ← Deskewing, denoising, 
         │  (magic-pdf)            │    shadow removal, normalization
         └────────────┬────────────┘
                      │
         ┌────────────┴────────────┐
         │ DeepDoctection Layout   │  ← Detect tables, forms,
         │  Detection              │    signatures, handwriting
         └────────────┬────────────┘
                      │
         ┌────────────┴────────────┐
         │   Hybrid OCR Engine     │
         │  • PaddleOCR (ML/EN)   │  ← Printed text
         │  • TrOCR (Handwritten) │  ← Handwritten text
         │  • Tesseract (Fallback)│  ← Legacy support
         └────────────┬────────────┘
                      │
         ┌────────────┴────────────┐
         │  Office Processor       │  ← Extract from DOCX/
         │  (textract + libraries) │    XLSX/PPTX directly
         └────────────┬────────────┘
                      │
         ┌────────────┴────────────┐
         │  Qwen-VL Vision         │  ← Analyze images,
         │  Analysis               │    diagrams, charts
         └────────────┬────────────┘
                      │
         ┌────────────┴────────────┐
         │  Docling Normalizer     │  ← Convert to canonical
         │  (Structured Output)    │    JSON/Markdown format
         └────────────┬────────────┘
                      │
         ┌────────────┴────────────┐
         │  Reasoning Engine       │  ← GPT-4o/Qwen2 for
         │  (GPT-4o / Qwen2)      │    insights & summarization
         └────────────┬────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│             STRUCTURED OUTPUT + ANALYSIS                     │
│  • JSON/Markdown with complete extraction                   │
│  • Executive summary & key insights                         │
│  • Action items & risk highlights                           │
│  • Department routing suggestions                           │
└─────────────────────────────────────────────────────────────┘
```

## Features

### 🔍 Comprehensive Document Support
- **PDFs**: Both digitally-born and scanned documents
- **Office Files**: Word, Excel, PowerPoint with table/image extraction
- **Images**: JPG, PNG, TIFF from phones, scanners, cameras
- **Multilingual**: Malayalam and English OCR support
- **Handwriting**: Recognition of handwritten notes

### 🎯 Advanced Processing
- **Layout Detection**: Tables, forms, signatures, stamps, diagrams
- **Preprocessing**: Automatic deskewing, denoising, shadow removal
- **Hybrid OCR**: Best-of-breed engines for different content types
- **Vision Analysis**: AI-powered image and diagram descriptions
- **Reasoning**: LLM-based insights, summaries, and recommendations

### 📊 Structured Output
- **JSON Format**: Machine-readable structured data
- **Markdown Format**: Human-readable formatted text
- **Metadata**: Processing info, confidence scores, timestamps
- **Analysis**: Insights, action items, routing suggestions

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd sih-doc
```

### 2. Install System Dependencies

#### Tesseract OCR
**Windows:**
```powershell
# Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH: C:\Program Files\Tesseract-OCR
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-mal tesseract-ocr-eng
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

#### PaddleOCR Models
Malayalam and English models will be downloaded automatically on first use.

### 3. Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# For GPU support (recommended):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Configure Environment

Copy `.env.example` to `.env` and configure:

```bash
copy .env.example .env
```

Edit `.env`:
```env
# Required for reasoning (choose one)
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# GPU acceleration (set to false if no GPU)
USE_GPU=true

# OCR languages
OCR_LANGUAGES=en,ml

# Processing options
ENABLE_PREPROCESSING=true
ENABLE_VISION_ANALYSIS=true
ENABLE_REASONING=true

# Model selections
VISION_MODEL=Qwen/Qwen2-VL-7B-Instruct
REASONING_MODEL=gpt-4o
HANDWRITING_MODEL=microsoft/trocr-large-handwritten
```

## Usage

### Command Line

#### Process Single File
```bash
python pipeline.py input_files/document.pdf --output output --format both
```

#### Process Directory
```bash
python pipeline.py input_files/ --output output --format json
```

### Python API

```python
from pathlib import Path
from pipeline import HybridDocumentPipeline

# Initialize pipeline
pipeline = HybridDocumentPipeline()

# Process single document
result = pipeline.process_document(Path("document.pdf"))

if result["success"]:
    print(f"Text: {result['structured_document']['content']['text']}")
    print(f"Summary: {result['structured_document']['analysis']['summary']}")
    print(f"Output: {result['output_paths']}")

# Process multiple documents
results = pipeline.process_batch([
    Path("doc1.pdf"),
    Path("doc2.docx"),
    Path("image.jpg")
])

for result in results:
    if result["success"]:
        print(f"✅ {result['file_path']}")
    else:
        print(f"❌ {result['file_path']}: {result['error']}")
```

### Custom Configuration

```python
from config import PipelineConfig, OCRConfig, ModelConfig

# Create custom config
custom_config = PipelineConfig(
    ocr=OCRConfig(
        languages=["en", "ml"],
        primary_engine="paddle"
    ),
    models=ModelConfig(
        reasoning_model="gpt-4o-mini",
        vision_model="Qwen/Qwen2-VL-2B-Instruct"
    )
)

# Use custom config
pipeline = HybridDocumentPipeline(custom_config)
```

## Output Format

### JSON Output
```json
{
  "document_info": {
    "file_name": "example.pdf",
    "file_type": "pdf",
    "file_size": 1024000,
    "processing_timestamp": "2025-05-15T10:30:00"
  },
  "content": {
    "text": "Full extracted text...",
    "structured_data": {
      "tables": [...],
      "images": [...]
    }
  },
  "layout": {
    "regions": [
      {
        "type": "text",
        "bbox": [100, 200, 500, 300],
        "content": "Text content...",
        "confidence": 0.95
      }
    ]
  },
  "analysis": {
    "summary": "Executive summary...",
    "key_insights": [...],
    "action_items": [...],
    "risk_highlights": [...],
    "routing_suggestions": {"departments": [...]}
  }
}
```

### Markdown Output
```markdown
# Document: example.pdf

**Processing Date**: 2025-05-15 10:30:00
**Document Type**: PDF
**Pages**: 10

## Summary

Executive summary of the document content...

## Key Insights

- Important point 1
- Important point 2
- Important point 3

## Content

Full extracted text with preserved structure...

## Tables

| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |

## Images & Diagrams

### Image 1: Technical Diagram
Description: This diagram shows...
```

## Configuration

All settings are managed via `.env` file and `config.py`:

| Setting | Description | Default |
|---------|-------------|---------|
| `USE_GPU` | Enable GPU acceleration | `false` |
| `OCR_LANGUAGES` | Languages for OCR | `en,ml` |
| `OCR_PRIMARY_ENGINE` | Primary OCR engine | `paddle` |
| `ENABLE_PREPROCESSING` | Enable MinerU preprocessing | `true` |
| `ENABLE_VISION_ANALYSIS` | Enable Qwen-VL analysis | `true` |
| `ENABLE_REASONING` | Enable LLM reasoning | `true` |
| `VISION_MODEL` | Vision model to use | `Qwen/Qwen2-VL-7B-Instruct` |
| `REASONING_MODEL` | LLM for reasoning | `gpt-4o` |
| `OUTPUT_FORMAT` | Output format | `both` |

## Supported File Types

| Category | Extensions | Processing Method |
|----------|-----------|-------------------|
| PDF | `.pdf` | MinerU → Layout → OCR → Vision |
| Word | `.docx`, `.doc` | textract + python-docx |
| Excel | `.xlsx`, `.xls` | textract + openpyxl |
| PowerPoint | `.pptx`, `.ppt` | textract + python-pptx |
| Images | `.jpg`, `.png`, `.jpeg`, `.tiff`, `.bmp` | Preprocessing → Layout → OCR → Vision |

## Performance

### Processing Times (Approximate)
- **PDF (10 pages)**: ~30-60 seconds
- **Office Document**: ~5-15 seconds
- **Image**: ~10-30 seconds

### Optimization Tips
1. **Enable GPU**: Set `USE_GPU=true` for 3-5x speedup
2. **Disable Optional Features**: Turn off vision/reasoning if not needed
3. **Batch Processing**: Process multiple files together
4. **Image Quality**: Higher DPI = better OCR but slower processing

## Malayalam Language Support

The pipeline has native Malayalam support:

1. **PaddleOCR**: Primary engine for Malayalam printed text
2. **Tesseract**: Fallback with `mal` language pack
3. **TrOCR**: Handwritten Malayalam recognition
4. **Language Detection**: Automatic detection of Malayalam vs English

### Testing Malayalam
```bash
# Process Malayalam document
python pipeline.py malayalam_doc.pdf --format both
```

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**GPU Not Detected**
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CPU version:
pip install torch torchvision torchaudio
```

**Tesseract Not Found**
```bash
# Windows: Add to PATH
$env:PATH += ";C:\Program Files\Tesseract-OCR"

# Linux: Install
sudo apt-get install tesseract-ocr tesseract-ocr-mal
```

**Model Download Issues**
```bash
# Set HuggingFace cache location
$env:HF_HOME = "C:\path\to\cache"

# Download models manually
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/trocr-large-handwritten')"
```

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Development

### Project Structure
```
sih-doc/
├── config.py              # Configuration management
├── pipeline.py            # Main orchestrator
├── preprocessor.py        # MinerU preprocessing
├── layout_analyzer.py     # DeepDoctection layout
├── ocr_engine.py         # Hybrid OCR system
├── office_processor.py   # Office document extraction
├── normalizer.py         # Docling normalization
├── vision_analyzer.py    # Qwen-VL vision analysis
├── reasoning_engine.py   # LLM reasoning
├── requirements.txt      # Dependencies
├── .env.example          # Configuration template
├── README.md             # This file
└── input_files/          # Input documents
```

### Adding New Features

1. **New OCR Engine**: Extend `HybridOCREngine` in `ocr_engine.py`
2. **New File Type**: Add handler in `pipeline.py`
3. **Custom Analysis**: Extend `ReasoningEngine` in `reasoning_engine.py`

## Acknowledgments

Built with:
- [MinerU](https://github.com/opendatalab/MinerU) - PDF preprocessing
- [DeepDoctection](https://github.com/deepdoctection/deepdoctection) - Layout analysis
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - Malayalam/English OCR
- [TrOCR](https://huggingface.co/microsoft/trocr-large-handwritten) - Handwriting recognition
- [Docling](https://github.com/DS4SD/docling) - Document normalization
- [Qwen-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) - Vision-language analysis
- [OpenAI GPT-4o](https://openai.com) - Reasoning and summarization
