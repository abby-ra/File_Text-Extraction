from paddleocr import PaddleOCR
import cv2

# Initialize PaddleOCR
ocr = PaddleOCR(use_textline_orientation=False, lang='en')

# Read image
image_path = r"input_files\text_image.jpg"
image = cv2.imread(image_path)

print(f"Image shape: {image.shape}")

# Run OCR
result = ocr.ocr(image)

print(f"\nResult type: {type(result)}")
print(f"Result length: {len(result)}")
print(f"result[0] type: {type(result[0])}")
print(f"result[0]: {result[0]}")

if result and result[0]:
    print(f"\nNumber of text lines detected: {len(result[0])}")
    for idx, line in enumerate(result[0]):
        print(f"\nLine {idx}:")
        print(f"  Structure: {line}")
        print(f"  Type: {type(line)}")
        if line and len(line) >= 2:
            print(f"  Bbox: {line[0]}")
            print(f"  Text info: {line[1]}")
            if isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                print(f"  Text: '{line[1][0]}'")
                print(f"  Confidence: {line[1][1]}")
