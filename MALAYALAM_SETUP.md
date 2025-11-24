# Malayalam OCR Setup Guide

## Current Situation

After testing multiple Malayalam OCR solutions, here's what we found:

### ❌ Not Available
1. **Chithrakan** - Not available via pip, GitHub repo may not be public
2. **EasyOCR** - Downloaded but Malayalam ('ml') language not supported
3. **PaddleOCR** - No Malayalam language models available

### ✅ Working Solution: Tesseract OCR

**Tesseract OCR** is the most reliable option for Malayalam text extraction.

## Installation Steps

### Step 1: Download Tesseract
Download the Windows installer from:
https://github.com/UB-Mannheim/tesseract/wiki

**Recommended version**: tesseract-ocr-w64-setup-5.3.3.20231005.exe

### Step 2: Install Tesseract
1. Run the installer
2. During installation, make sure to select:
   - ✅ **Additional language data (download)**
   - ✅ **Malayalam** language pack
3. Note the installation path (usually `C:\Program Files\Tesseract-OCR`)

### Step 3: Add to System PATH
1. Open System Properties → Environment Variables
2. Edit the `Path` variable
3. Add: `C:\Program Files\Tesseract-OCR`
4. Click OK and restart PowerShell

### Step 4: Verify Installation
```powershell
# Check Tesseract is installed
tesseract --version

# Check Malayalam language is available
tesseract --list-langs | Select-String "mal"
```

### Step 5: Install Python Package
```powershell
pip install pytesseract
```

## Testing Malayalam Extraction

Once Tesseract is installed, your extraction pipeline will automatically:

1. ✅ Use **Tesseract** for Malayalam text (primary engine)
2. ✅ Use **PaddleOCR** for English text
3. ✅ Use **TrOCR** for handwritten text
4. ✅ Preserve Malayalam script in UTF-8 output

## Example Usage

```powershell
# Place your Malayalam files in input_files/
# Then run:
python enhanced_extract.py

# The output will show:
# [Malayalam+English - Tesseract]
# മലയാളം ടെക്സ്റ്റ് ഇവിടെ...
```

## Alternative: Manual Malayalam Support

If you can't install Tesseract, you can use these online services:
1. **Google Cloud Vision API** (paid, excellent Malayalam support)
2. **Microsoft Azure Computer Vision** (paid, good Indic language support)
3. **AI4Bharat IndicOCR** (if available)

## Current Pipeline Status

**Working Now:**
- ✅ English OCR (PaddleOCR)
- ✅ Handwriting OCR (TrOCR)
- ✅ All file formats (PDF, DOCX, images, etc.)
- ✅ Structured JSON output (Docling)

**Needs Tesseract for:**
- ⚠️ Malayalam text extraction
- ⚠️ Mixed script documents

## Quick Test

After installing Tesseract, test it with:

```powershell
# Test Tesseract directly
tesseract malayalam_image.png output -l mal

# Or use the full pipeline
python enhanced_extract.py
```

Your Malayalam text will be extracted and saved to the output folder!

---

**Note**: The enhanced_extract.py is already configured to use Tesseract as the Malayalam OCR engine. You just need to install Tesseract on your system.
