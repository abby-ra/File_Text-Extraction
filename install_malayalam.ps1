# Installation Script for Malayalam OCR Support

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Enhanced OCR Pipeline - Malayalam Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "[INFO] Activating virtual environment..." -ForegroundColor Yellow
    & .\.venv\Scripts\Activate.ps1
}

Write-Host "[STEP 1] Installing core dependencies..." -ForegroundColor Green
python -m pip install --upgrade pip
pip install -r requirements_enhanced.txt

Write-Host ""
Write-Host "[STEP 2] Installing Malayalam OCR - Chithrakan..." -ForegroundColor Green

# Try pip install first
$chithrakanInstalled = $false
try {
    pip install chithrakan 2>&1 | Out-Null
    $chithrakanInstalled = $true
    Write-Host "  [OK] Chithrakan installed via pip" -ForegroundColor Green
} catch {
    Write-Host "  [WARNING] Chithrakan not available via pip" -ForegroundColor Yellow
}

# If pip install failed, try alternative method
if (-not $chithrakanInstalled) {
    Write-Host "  [INFO] Attempting to install from GitHub..." -ForegroundColor Yellow
    
    if (Test-Path "temp_chithrakan") {
        Remove-Item -Recurse -Force "temp_chithrakan"
    }
    
    try {
        git clone https://github.com/AI4Bharat/Chithrakan.git temp_chithrakan
        Set-Location temp_chithrakan
        pip install -e .
        Set-Location ..
        Remove-Item -Recurse -Force "temp_chithrakan"
        Write-Host "  [OK] Chithrakan installed from source" -ForegroundColor Green
        $chithrakanInstalled = $true
    } catch {
        Write-Host "  [ERROR] Failed to install Chithrakan from GitHub" -ForegroundColor Red
        Write-Host "          Malayalam OCR will not be available" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "[STEP 3] Checking Tesseract installation..." -ForegroundColor Green

try {
    tesseract --version 2>&1 | Out-Null
    Write-Host "  [OK] Tesseract is installed" -ForegroundColor Green
    
    # Check for Malayalam language data
    $tessdata = tesseract --list-langs 2>&1 | Select-String "mal"
    if ($tessdata) {
        Write-Host "  [OK] Malayalam language data found" -ForegroundColor Green
    } else {
        Write-Host "  [WARNING] Malayalam language data not found" -ForegroundColor Yellow
        Write-Host "            Download from: https://github.com/tesseract-ocr/tessdata" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  [WARNING] Tesseract not found in PATH" -ForegroundColor Yellow
    Write-Host "            Download from: https://github.com/UB-Mannheim/tesseract/wiki" -ForegroundColor Yellow
    Write-Host "            Tesseract fallback will not be available" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[STEP 4] Verifying installation..." -ForegroundColor Green

python -c @"
import sys
print('  [INFO] Checking installed packages...')

packages = {
    'paddleocr': 'PaddleOCR',
    'cv2': 'OpenCV',
    'fitz': 'PyMuPDF',
    'docx': 'python-docx',
    'transformers': 'Transformers',
    'docling': 'Docling'
}

for module, name in packages.items():
    try:
        __import__(module)
        print(f'  [OK] {name}')
    except ImportError:
        print(f'  [ERROR] {name} not installed')

# Check optional packages
try:
    import chithrakan
    print('  [OK] Chithrakan (Malayalam OCR)')
except ImportError:
    print('  [WARNING] Chithrakan not available')

try:
    import pytesseract
    print('  [OK] PyTesseract')
except ImportError:
    print('  [WARNING] PyTesseract not available')
"@

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installation Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($chithrakanInstalled) {
    Write-Host "✓ Malayalam OCR (Chithrakan): READY" -ForegroundColor Green
} else {
    Write-Host "✗ Malayalam OCR (Chithrakan): NOT AVAILABLE" -ForegroundColor Red
    Write-Host "  Note: You can still use Tesseract for Malayalam" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Place your files in the 'input_files' folder" -ForegroundColor White
Write-Host "2. Run: python enhanced_extract.py" -ForegroundColor White
Write-Host "3. Check results in 'output' folder" -ForegroundColor White
Write-Host ""
Write-Host "For Malayalam support without Chithrakan:" -ForegroundColor Yellow
Write-Host "- Install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki" -ForegroundColor Yellow
Write-Host "- Download Malayalam language data (mal.traineddata)" -ForegroundColor Yellow
Write-Host ""
