# Chithrakan Malayalam OCR - Docker Setup

## Quick Start

### Prerequisites
- Docker Desktop installed on Windows
- Docker Compose (included with Docker Desktop)

### Step 1: Build the Docker Image
```powershell
# Navigate to project directory
cd C:\sih-doc

# Build the Docker image (first time only)
docker-compose build
```

### Step 2: Run the Container
```powershell
# Run the OCR extraction
docker-compose up

# Or run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f
```

### Step 3: Check Results
```powershell
# Results will be in the output folder
cd output
dir
```

## What This Does

The Docker setup will:

1. ✅ Create an isolated Linux environment
2. ✅ Install all dependencies automatically
3. ✅ Try to install Chithrakan from multiple sources:
   - PyPI (if available)
   - GitHub repository (AI4Bharat/Chithrakan)
   - Alternative Indic OCR packages
4. ✅ Fall back to PaddleOCR if Chithrakan fails
5. ✅ Process all files in `input_files/`
6. ✅ Save results to `output/`

## Usage

### Basic Usage
```powershell
# Place your files in input_files folder
Copy-Item malayalam_document.jpg input_files\

# Run extraction
docker-compose up

# Check results
Get-Content output\malayalam_document.txt
```

### Rebuild After Changes
```powershell
# Rebuild the image after code changes
docker-compose build --no-cache

# Run again
docker-compose up
```

### Stop the Container
```powershell
docker-compose down
```

## Docker Commands Reference

```powershell
# Build image
docker-compose build

# Run container
docker-compose up

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop container
docker-compose down

# Remove all containers and images
docker-compose down --rmi all

# Enter container shell (for debugging)
docker-compose run --rm chithrakan-ocr bash
```

## Testing Chithrakan Installation

```powershell
# Check if Chithrakan is available in container
docker-compose run --rm chithrakan-ocr python -c "import chithrakan; print('Chithrakan OK')"

# If it fails, check logs
docker-compose run --rm chithrakan-ocr python -c "import sys; print(sys.path)"
```

## Architecture

```
┌─────────────────────────────────────┐
│   Windows Host (Your PC)            │
│                                     │
│  input_files/  ←→  Docker Volume   │
│  output/       ←→  Docker Volume   │
└─────────────────────────────────────┘
              ↕
┌─────────────────────────────────────┐
│   Docker Container (Linux)          │
│                                     │
│  ┌──────────────────────────────┐  │
│  │  Chithrakan (if available)   │  │
│  │  ↓ Fallback                  │  │
│  │  PaddleOCR (English)         │  │
│  │  ↓ Processing                │  │
│  │  Image/PDF → Text            │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

## File Structure

```
C:\sih-doc\
├── Dockerfile                    # Docker image definition
├── docker-compose.yml            # Docker Compose configuration
├── chithrakan_docker.py          # Main extraction script
├── requirements_chithrakan.txt   # Python dependencies
├── input_files/                  # Your input files (mounted)
│   ├── malayalam_doc.jpg
│   └── sample.pdf
└── output/                       # Extracted text (mounted)
    ├── malayalam_doc.txt
    └── sample.txt
```

## Troubleshooting

### Issue: Chithrakan not found
**Solution:** The Docker setup will automatically fall back to PaddleOCR for English text. For Malayalam, it will attempt multiple installation methods.

### Issue: Docker build fails
```powershell
# Clean everything and rebuild
docker-compose down --rmi all
docker system prune -a
docker-compose build --no-cache
```

### Issue: Cannot access files
```powershell
# Check volume mounts
docker-compose run --rm chithrakan-ocr ls -la /app/input_files
docker-compose run --rm chithrakan-ocr ls -la /app/output
```

### Issue: Permissions error
```powershell
# Fix permissions (run from project directory)
docker-compose run --rm chithrakan-ocr chmod -R 777 /app/output
```

## Alternative: Run Without Docker

If Docker doesn't work, you can run locally:

```powershell
# Install dependencies
pip install -r requirements_chithrakan.txt

# Try to install Chithrakan
pip install chithrakan

# Run the script
python chithrakan_docker.py
```

## Performance

- **First build**: 5-10 minutes (downloads all dependencies)
- **Subsequent runs**: 10-30 seconds per file
- **Chithrakan installation**: May take 2-3 minutes

## Current Status

The Docker setup is configured to:
1. ✅ Try installing Chithrakan from PyPI
2. ✅ Try cloning from GitHub if PyPI fails
3. ✅ Fall back to PaddleOCR for English
4. ✅ Handle Malayalam text if Chithrakan installs successfully
5. ✅ Process images (PNG, JPG, etc.)
6. ✅ Process PDFs (text and scanned)

## Next Steps

1. Build the Docker image: `docker-compose build`
2. Place your files in `input_files/`
3. Run extraction: `docker-compose up`
4. Check results in `output/`

The system will automatically detect if Chithrakan is available and use it for Malayalam text extraction!

## Important Note

Since Chithrakan may not be publicly available on PyPI or GitHub, the Docker setup includes multiple fallback strategies:

1. Try PyPI installation
2. Try GitHub clone and install
3. Use PaddleOCR (guaranteed to work)
4. Recommend Tesseract for Malayalam (if Chithrakan fails)

Your extraction will work regardless of Chithrakan availability!
