import cv2
from ocr_engine import HybridOCREngine
from config import PipelineConfig

# Load config
config = PipelineConfig.from_env()

# Initialize OCR
ocr = HybridOCREngine(config)

# Load image
image = cv2.imread(r"input_files\text_image.jpg")
print(f"Image shape: {image.shape}")

# Run OCR on full image
result = ocr.extract_text(image, region_type="text", language="en")

print(f"\nOCR Result:")
print(f"Success: {result.get('success', False)}")
print(f"Engine: {result.get('engine', 'unknown')}")
print(f"Confidence: {result.get('confidence', 0)}")
print(f"Text length: {len(result.get('text', ''))}")
print(f"\nExtracted text:")
print(result.get('text', ''))[:500]  # First 500 chars
