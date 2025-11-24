# Qwen-VL Image Description Setup

## Overview
Qwen-VL is integrated into the extraction pipeline to provide detailed image descriptions, object detection, and visual analysis.

## Current Status
✅ Dependencies installed (transformers, torch, qwen-vl-utils)
✅ Integration code added to `enhanced_extract.py`
✅ Standalone module created (`qwen_image_describer.py`)
⏳ Model download in progress (~4GB - Qwen2-VL-2B-Instruct)

## Features

### 1. **Automatic Image Description**
When processing images, the pipeline will:
- Generate detailed descriptions of image content
- Identify objects, scenes, and context
- Extract visible text using vision-language understanding
- Describe colors, layout, and composition

### 2. **Multiple Use Cases**
```python
from qwen_image_describer import QwenImageDescriber

describer = QwenImageDescriber()

# General description
description = describer.describe_image("path/to/image.jpg")

# Ask specific questions
answer = describer.describe_image_with_question(
    "path/to/image.jpg", 
    "What objects are visible?"
)

# Extract text using VLM
text = describer.extract_text_from_image_vlm("path/to/document.png")

# Analyze documents
analysis = describer.analyze_document("path/to/form.jpg")
```

## Integration in Enhanced Extract

The extraction pipeline automatically uses Qwen-VL when processing images:

1. **Image Description** - Generated first for context
2. **OCR Text Extraction** - PaddleOCR, EasyOCR, TrOCR
3. **Separate Description Files** - Saved as `{filename}_description.txt`

### Output Example
For `photo.jpg`, you'll get:
- `photo.txt` - OCR extracted text
- `photo_description.txt` - Qwen-VL detailed description
- `photo.json` - Structured data (if applicable)

## Model Download

### Complete the Download
The Qwen2-VL-2B-Instruct model download was interrupted. To resume:

```powershell
python qwen_image_describer.py
```

The model will resume downloading from where it stopped.

### Alternative: Smaller Model
If the 2B model is too large, you can use a smaller version:

Edit `qwen_image_describer.py` line 21:
```python
# Change from:
model_name="Qwen/Qwen2-VL-2B-Instruct"

# To smaller model (if available):
model_name="Qwen/Qwen2-VL-500M-Instruct"  # Example smaller version
```

## Usage

### With Enhanced Extract
```powershell
python enhanced_extract.py
```

The pipeline will:
1. Check if Qwen-VL is available
2. Generate descriptions for images (if available)
3. Continue with OCR extraction regardless
4. Save both description and OCR text

### Standalone Testing
```powershell
python qwen_image_describer.py
```

Tests Qwen-VL with sample images from the workspace.

## Graceful Fallback

The system is designed to work even without Qwen-VL:
- If model not available: Skips description, continues with OCR
- If model fails: Logs warning, continues processing
- No interruption to the extraction pipeline

## Performance Notes

### CPU vs GPU
- **CPU**: Slower (30-60 seconds per image) but works everywhere
- **GPU**: Much faster (1-3 seconds per image) if CUDA available

### Memory Requirements
- Qwen2-VL-2B: ~4GB model + 2-4GB runtime memory
- Recommended: 8GB+ RAM for smooth operation

## Benefits of VLM vs Traditional OCR

| Feature | Traditional OCR | Qwen-VL |
|---------|----------------|---------|
| Text extraction | ✅ Good | ✅ Good |
| Context understanding | ❌ None | ✅ Excellent |
| Object detection | ❌ No | ✅ Yes |
| Scene description | ❌ No | ✅ Detailed |
| Multi-language | Limited | ✅ Better |
| Handwriting | Limited | ✅ Better |
| Visual reasoning | ❌ No | ✅ Yes |

## Troubleshooting

### Download Interrupted
The model download is resumed automatically. Just run the script again:
```powershell
python qwen_image_describer.py
```

### Out of Memory
If you get memory errors:
1. Close other applications
2. Use a smaller model variant
3. Process images one at a time
4. Consider using GPU if available

### Model Not Loading
Check the error in the console. Common issues:
- Insufficient disk space (~4GB needed)
- Network interruption (resume download)
- Missing dependencies (reinstall: `pip install transformers torch`)

## Example Output

### Input: `job_card.jpg`

**OCR Text (`job_card.txt`):**
```
SERVICE REPORT
Date: 23-11-2025
Vehicle: KL-01-AB-1234
Service Type: Regular Maintenance
```

**VLM Description (`job_card_description.txt`):**
```
Image Description (Qwen-VL)
======================================================================

This is a service job card document. The main elements visible are:

1. Header section with "SERVICE REPORT" title in bold
2. A table containing service details including date (23-11-2025), 
   vehicle registration number (KL-01-AB-1234), and service type
3. The document appears to be a printed form with handwritten notes
4. Blue company logo in the top right corner
5. Layout is professional with clear sections and borders
6. The paper appears slightly aged with minor stains near the edges
```

## Next Steps

1. **Complete model download** - Run `python qwen_image_describer.py`
2. **Test with your images** - Run `python enhanced_extract.py`
3. **Review descriptions** - Check `*_description.txt` files in output folder
4. **Fine-tune prompts** - Edit prompts in `qwen_image_describer.py` if needed

## Questions?

The system works with or without Qwen-VL. The VLM adds rich context and understanding but is not required for basic text extraction.
