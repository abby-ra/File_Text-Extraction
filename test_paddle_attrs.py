import cv2
from paddleocr import PaddleOCR

# Initialize
ocr = PaddleOCR(use_textline_orientation=False, lang='en')

# Load and OCR
image = cv2.imread(r"input_files\text_image.jpg")
result = ocr.ocr(image)

print(f"Result type: {type(result)}")
print(f"Result length: {len(result)}")
print(f"result[0] type: {type(result[0])}")
print(f"\nAvailable attributes:")
print(dir(result[0]))
print(f"\nHas rec_texts: {hasattr(result[0], 'rec_texts')}")
print(f"Has rec_scores: {hasattr(result[0], 'rec_scores')}")

if hasattr(result[0], 'rec_texts'):
    print(f"\nrec_texts: {result[0].rec_texts[:3]}")  # First 3
if hasattr(result[0], 'rec_scores'):
    print(f"rec_scores: {result[0].rec_scores[:3]}")  # First 3

print(f"\nKeys in result[0]:")
print(list(result[0].keys()))

print(f"\nChecking for text fields:")
for key in result[0].keys():
    if 'text' in key.lower() or 'rec' in key.lower():
        print(f"  {key}: {type(result[0][key])}")
        if isinstance(result[0][key], list) and len(result[0][key]) > 0:
            print(f"    First item: {result[0][key][0]}")
