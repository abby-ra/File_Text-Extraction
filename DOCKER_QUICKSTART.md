# Quick Start Guide - Chithrakan Docker Setup

## ⚠️ Important: Start Docker Desktop First!

Before running any docker commands:
1. Open **Docker Desktop** application
2. Wait for it to fully start (green indicator at bottom)
3. Then proceed with the commands below

## 🚀 Quick Setup (3 Steps)

### Step 1: Ensure Docker Desktop is Running
```powershell
# Check if Docker is running
docker --version

# If you see an error about "docker daemon", start Docker Desktop first!
```

### Step 2: Build the Docker Image
```powershell
# This will take 5-10 minutes (first time only)
docker-compose build
```

### Step 3: Run Extraction
```powershell
# Run the OCR extraction
docker-compose up

# Check results
cd output
dir *.txt
```

## 📋 Complete Workflow

```powershell
# 1. Start Docker Desktop (wait for it to be ready)

# 2. Navigate to project
cd C:\sih-doc

# 3. Build image (first time only)
docker-compose build

# 4. Add your files to input_files/
Copy-Item C:\path\to\malayalam.jpg .\input_files\

# 5. Run extraction
docker-compose up

# 6. View results
Get-Content .\output\malayalam.txt
```

## 🔧 What Gets Installed in Docker

The Docker container will automatically try:
1. **Chithrakan** from PyPI
2. **Chithrakan** from GitHub (AI4Bharat/Chithrakan)  
3. **PaddleOCR** (fallback for English)
4. **All dependencies** automatically

## ✅ Expected Output

When successful, you'll see:
```
[OK] Chithrakan loaded successfully!
Initializing OCR engines...
[OK] English PaddleOCR ready
======================================================================
Chithrakan Docker-based Malayalam OCR
Chithrakan available: True
======================================================================

Processing: malayalam_image.jpg
  Trying Chithrakan...
  [SUCCESS] Saved to: malayalam_image.txt
  Engines used: Chithrakan
```

## ⚠️ If Chithrakan is Not Available

The system will automatically fall back:
```
[WARNING] Chithrakan not available, using fallback OCR engines
Initializing OCR engines...
[OK] English PaddleOCR ready

Processing: malayalam_image.jpg
  Trying PaddleOCR...
  [SUCCESS] Saved to: malayalam_image.txt
  Engines used: PaddleOCR
```

## 🐛 Troubleshooting

### Error: "cannot find file specified" or "docker daemon"
**Solution:** Docker Desktop is not running
```powershell
# Start Docker Desktop application
# Wait for green indicator at bottom
# Then try again
```

### Error: "version is obsolete"
**Solution:** Already fixed - using latest docker-compose syntax

### Build takes too long
**Normal:** First build takes 5-10 minutes
- Downloads Python base image (~100MB)
- Installs all dependencies (~500MB)
- Attempts Chithrakan installation

### How to rebuild after changes
```powershell
# Rebuild without cache
docker-compose build --no-cache

# Run again
docker-compose up
```

### View detailed logs
```powershell
# See what's happening inside container
docker-compose logs -f
```

### Enter container for debugging
```powershell
# Get a shell inside the container
docker-compose run --rm chithrakan-ocr bash

# Then test manually:
python -c "import chithrakan; print('OK')"
ls -la /app/input_files
```

## 📁 File Structure

```
C:\sih-doc\
├── Dockerfile                    ← Container definition
├── docker-compose.yml            ← Orchestration config
├── chithrakan_docker.py          ← Extraction script
├── requirements_chithrakan.txt   ← Dependencies
├── input_files/                  ← YOUR FILES HERE
│   └── *.jpg, *.png, *.pdf
└── output/                       ← RESULTS HERE
    └── *.txt
```

## 🎯 Advantages of Docker Approach

✅ **Isolated environment** - No conflicts with system packages
✅ **Automatic setup** - Installs everything for you
✅ **Cross-platform** - Works on Windows, Mac, Linux
✅ **Reproducible** - Same environment every time
✅ **Easy cleanup** - Just delete container

## 🧹 Cleanup

```powershell
# Stop and remove container
docker-compose down

# Remove everything including images
docker-compose down --rmi all

# Remove all Docker data (careful!)
docker system prune -a
```

## 📝 Current Status

**Chithrakan Status:** Will be attempted during Docker build
- If available: ✅ Malayalam OCR ready
- If not available: ⚠️ Falls back to PaddleOCR (English only)

**Recommendation:** Since Chithrakan may not be publicly available, also consider:
- Installing Tesseract (see MALAYALAM_SETUP.md)
- Using enhanced_extract.py with Tesseract
- Google Cloud Vision API for production use

## 🔄 Alternative Without Docker

If Docker doesn't work, use the non-Docker approach:
```powershell
# Use the enhanced extraction with Tesseract
python enhanced_extract.py
```

See `MALAYALAM_SETUP.md` for Tesseract installation.

---

**Next Steps:**
1. ✅ Start Docker Desktop
2. ✅ Run `docker-compose build`
3. ✅ Place files in `input_files/`
4. ✅ Run `docker-compose up`
5. ✅ Check results in `output/`
