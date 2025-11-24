"""
Chithrakan Docker-based Malayalam OCR Extraction
Fallback to available OCR engines if Chithrakan is not available
"""
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import fitz
from PIL import Image

# Try importing Chithrakan
chithrakan_available = False
try:
    import chithrakan
    chithrakan_available = True
    print("[OK] Chithrakan loaded successfully!")
except ImportError:
    print("[WARNING] Chithrakan not available, using fallback OCR engines")

# Import PaddleOCR
from paddleocr import PaddleOCR

# Initialize OCR engines
print("Initializing OCR engines...")
ocr_en = PaddleOCR(lang='en', use_textline_orientation=False, show_log=False)
print("[OK] English PaddleOCR ready")

def extract_with_chithrakan(image_path):
    """Extract text using Chithrakan if available"""
    if not chithrakan_available:
        return None
    
    try:
        # Chithrakan usage (adjust based on actual API)
        # This is a generic implementation - adjust when real API is known
        result = chithrakan.recognize(str(image_path))
        
        if isinstance(result, dict):
            texts = []
            if 'text' in result:
                texts.append(result['text'])
            elif 'lines' in result:
                texts.extend([line.get('text', '') for line in result['lines']])
            return '\n'.join(texts) if texts else None
        elif isinstance(result, str):
            return result
        
        return None
    except Exception as e:
        print(f"    Chithrakan error: {str(e)[:80]}")
        return None

def extract_with_paddleocr(image_path):
    """Extract text using PaddleOCR"""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        result = ocr_en.ocr(image, cls=False)
        if result and len(result) > 0 and result[0]:
            ocr_result = result[0]
            if hasattr(ocr_result, 'keys') and 'rec_texts' in ocr_result:
                texts = [str(t).strip() for t in ocr_result['rec_texts'] if str(t).strip()]
                return '\n'.join(texts) if texts else None
        
        return None
    except Exception as e:
        print(f"    PaddleOCR error: {str(e)[:80]}")
        return None

def process_image(image_path, output_dir):
    """Process a single image file"""
    image_path = Path(image_path)
    print(f"\nProcessing: {image_path.name}")
    
    all_texts = []
    engines_used = []
    
    # Try Chithrakan first for Malayalam
    if chithrakan_available:
        print("  Trying Chithrakan...")
        text = extract_with_chithrakan(image_path)
        if text and len(text.strip()) > 5:
            all_texts.append(f"[Malayalam - Chithrakan]\n{text}")
            engines_used.append("Chithrakan")
    
    # Try PaddleOCR for English
    if len(all_texts) == 0:
        print("  Trying PaddleOCR...")
        text = extract_with_paddleocr(image_path)
        if text:
            all_texts.append(f"[English - PaddleOCR]\n{text}")
            engines_used.append("PaddleOCR")
    
    # Save results
    output_path = output_dir / f"{image_path.stem}.txt"
    final_text = '\n\n'.join(all_texts) if all_texts else "[No text extracted]"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_text)
    
    print(f"  [SUCCESS] Saved to: {output_path.name}")
    print(f"  Engines used: {', '.join(engines_used) if engines_used else 'None'}")
    return True

def process_pdf(pdf_path, output_dir):
    """Process PDF file"""
    pdf_path = Path(pdf_path)
    print(f"\nProcessing: {pdf_path.name}")
    
    try:
        doc = fitz.open(pdf_path)
        all_text = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            if text.strip():
                all_text.append(f"[Page {page_num + 1}]\n{text}")
            else:
                # Convert page to image and OCR
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_bytes = pix.tobytes("png")
                
                # Save temp image
                temp_img = output_dir / f"temp_page_{page_num}.png"
                with open(temp_img, 'wb') as f:
                    f.write(img_bytes)
                
                # Extract text
                page_text = []
                if chithrakan_available:
                    text = extract_with_chithrakan(temp_img)
                    if text:
                        page_text.append(f"[Malayalam - Chithrakan]\n{text}")
                
                if not page_text:
                    text = extract_with_paddleocr(temp_img)
                    if text:
                        page_text.append(f"[English - PaddleOCR]\n{text}")
                
                if page_text:
                    all_text.append(f"[Page {page_num + 1} - OCR]\n" + '\n'.join(page_text))
                
                # Clean up temp file
                if temp_img.exists():
                    temp_img.unlink()
        
        doc.close()
        
        # Save results
        output_path = output_dir / f"{pdf_path.stem}.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(all_text) if all_text else "[No text extracted]")
        
        print(f"  [SUCCESS] Saved to: {output_path.name}")
        return True
    except Exception as e:
        print(f"  [ERROR] {str(e)[:100]}")
        return False

def main():
    """Main processing function"""
    input_dir = Path("/app/input_files")
    output_dir = Path("/app/output")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("Chithrakan Docker-based Malayalam OCR")
    print(f"Chithrakan available: {chithrakan_available}")
    print("="*70)
    
    # Get all files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]
    pdf_files = [f for f in input_dir.iterdir() if f.suffix.lower() == '.pdf']
    
    print(f"\nFound {len(image_files)} images and {len(pdf_files)} PDFs")
    
    successful = 0
    failed = 0
    
    # Process images
    for img_file in image_files:
        try:
            if process_image(img_file, output_dir):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [ERROR] {str(e)[:100]}")
            failed += 1
    
    # Process PDFs
    for pdf_file in pdf_files:
        try:
            if process_pdf(pdf_file, output_dir):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [ERROR] {str(e)[:100]}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"Processing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print("="*70)

if __name__ == "__main__":
    main()
