"""
Intelligent Text Extraction Tool
Automatically detects file types and extracts text using appropriate tools:
- PDF: PyMuPDF for text, Tesseract for images, Amazon Textract for tables
- Images: Tesseract OCR
- Text files: Direct reading
- Word docs: python-docx
Handles 500+ page documents efficiently.
"""

import os
os.environ['PATH'] += os.pathsep + r'C:\Program Files\Tesseract-OCR'

import os
import sys
import magic
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import boto3
from docx import Document
import json
import time
from pathlib import Path
import logging
from typing import List, Tuple, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntelligentTextExtractor:
    def __init__(self, input_folder="input_files", output_folder="extracted_texts"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.setup_folders()
        
        # Initialize AWS Textract client (you'll need to configure AWS credentials)
        try:
            self.textract_client = boto3.client('textract', region_name='us-east-1')
        except Exception as e:
            logger.warning(f"AWS Textract not configured: {e}")
            self.textract_client = None

    def setup_folders(self):
        """Create input and output folders if they don't exist"""
        os.makedirs(self.input_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
        logger.info(f"Created folders: {self.input_folder}, {self.output_folder}")

    def detect_file_type(self, file_path: str) -> str:
        """Detect file type using python-magic"""
        try:
            mime_type = magic.from_file(file_path, mime=True)
            logger.info(f"Detected MIME type for {file_path}: {mime_type}")
            
            if mime_type == 'application/pdf':
                return 'pdf'
            elif mime_type.startswith('image/'):
                return 'image'
            elif mime_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 
                              'application/msword']:
                return 'docx'
            elif mime_type.startswith('text/'):
                return 'text'
            else:
                return 'unknown'
        except Exception as e:
            logger.error(f"Error detecting file type: {e}")
            # Fallback to file extension
            ext = Path(file_path).suffix.lower()
            if ext == '.pdf':
                return 'pdf'
            elif ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                return 'image'
            elif ext in ['.docx', '.doc']:
                return 'docx'
            elif ext in ['.txt', '.text']:
                return 'text'
            return 'unknown'

    def analyze_pdf_content(self, pdf_path: str) -> Dict:
        """Analyze PDF to determine content types (text, images, tables)"""
        doc = fitz.open(pdf_path)
        analysis = {
            'has_text': False,
            'has_images': False,
            'has_tables': False,
            'total_pages': len(doc),
            'text_pages': [],
            'image_pages': [],
            'table_pages': []
        }
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Check for text
            text = page.get_text().strip()
            if text:
                analysis['has_text'] = True
                analysis['text_pages'].append(page_num)
            
            # Check for images
            image_list = page.get_images()
            if image_list:
                analysis['has_images'] = True
                analysis['image_pages'].append(page_num)
            
            # Simple table detection (look for multiple tabs or grid-like structure)
            if text and ('\t' in text or text.count('|') > 5):
                analysis['has_tables'] = True
                analysis['table_pages'].append(page_num)
        
        doc.close()
        logger.info(f"PDF analysis: {analysis}")
        return analysis

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF"""
        doc = fitz.open(pdf_path)
        text_content = []
        logger.info(f"Extracting text from {len(doc)} pages using PyMuPDF...")
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                text_content.append(f"--- Page {page_num + 1} ---\n{text}")
            # Progress indicator for large documents
            if (page_num + 1) % 50 == 0:
                logger.info(f"Processed {page_num + 1}/{len(doc)} pages")
        doc.close()
        result = "\n\n".join(text_content)
        # If no text found, fallback to OCR (Malayalam+English)
        if not result.strip():
            logger.info("No selectable text found, using Tesseract OCR for Malayalam/English PDF.")
            result = self.extract_images_from_pdf_ocr(pdf_path)
        return result

    def extract_images_from_pdf_ocr(self, pdf_path: str, page_numbers: List[int] = None) -> str:
        """Extract text from PDF images using Tesseract OCR"""
        doc = fitz.open(pdf_path)
        ocr_content = []
        
        pages_to_process = page_numbers or range(len(doc))
        logger.info(f"Extracting text from {len(pages_to_process)} pages using OCR...")
        
        for page_num in pages_to_process:
            page = doc[page_num]
            # Convert page to image
            mat = fitz.Matrix(2.0, 2.0)  # Higher resolution for better OCR
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            # OCR the image (Malayalam + English)
            try:
                image = Image.open(io.BytesIO(img_data))
                # Always use Malayalam+English model for best coverage
                ocr_text = pytesseract.image_to_string(image, lang='mal+eng')
                if ocr_text.strip():
                    ocr_content.append(f"--- Page {page_num + 1} (OCR) ---\n{ocr_text}")
            except Exception as e:
                logger.error(f"OCR failed for page {page_num + 1}: {e}")
            # Progress indicator
            if (page_num + 1) % 10 == 0:
                logger.info(f"OCR processed {page_num + 1} pages")
        
        doc.close()
        return "\n\n".join(ocr_content)

    def extract_tables_from_pdf_textract(self, pdf_path: str, page_numbers: List[int] = None) -> str:
        """Extract tables from PDF using Amazon Textract"""
        if not self.textract_client:
            logger.warning("Textract not available, skipping table extraction")
            return ""
        
        try:
            # For large PDFs, process in batches
            with open(pdf_path, 'rb') as document:
                response = self.textract_client.analyze_document(
                    Document={'Bytes': document.read()},
                    FeatureTypes=['TABLES']
                )
            
            # Extract table data
            table_content = []
            for block in response['Blocks']:
                if block['BlockType'] == 'TABLE':
                    table_text = self.parse_textract_table(block, response['Blocks'])
                    table_content.append(table_text)
            
            return "\n\n".join(table_content)
        
        except Exception as e:
            logger.error(f"Textract extraction failed: {e}")
            return ""

    def parse_textract_table(self, table_block: Dict, all_blocks: List[Dict]) -> str:
        """Parse Textract table response into readable text"""
        # Simplified table parsing - you may want to enhance this
        table_text = "--- TABLE ---\n"
        
        # This is a simplified implementation
        # In practice, you'd want to properly parse the table structure
        for relationship in table_block.get('Relationships', []):
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    child_block = next((b for b in all_blocks if b['Id'] == child_id), None)
                    if child_block and child_block['BlockType'] == 'CELL':
                        if 'Text' in child_block:
                            table_text += child_block['Text'] + " | "
        
        return table_text

    def extract_from_image(self, image_path: str) -> str:
        """Extract text from image using Tesseract OCR"""
        try:
            logger.info(f"Extracting text from image: {image_path}")
            image = Image.open(image_path)
            # Always use Malayalam+English model for best coverage
            text = pytesseract.image_to_string(image, lang='mal+eng')
            return text
        except Exception as e:
            logger.error(f"Image OCR failed: {e}")
            return ""

    def extract_from_docx(self, docx_path: str) -> str:
        """Extract text from Word document"""
        try:
            logger.info(f"Extracting text from Word document: {docx_path}")
            doc = Document(docx_path)
            text_content = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text)
            result = "\n".join(text_content)
            # If no text found, fallback to OCR (convert each page to image and OCR)
            if not result.strip():
                logger.info("No text found in DOCX, attempting OCR (Malayalam/English) on images (not implemented here)")
                # Optionally, implement DOCX to image conversion and OCR here
            return result
        except Exception as e:
            logger.error(f"Word document extraction failed: {e}")
            return ""

    def extract_from_text_file(self, text_path: str) -> str:
        """Extract text from plain text file"""
        try:
            logger.info(f"Reading text file: {text_path}")
            with open(text_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Text file reading failed: {e}")
            return ""

    def process_single_file(self, file_path: str) -> str:
        """Process a single file and extract all text content"""
        file_type = self.detect_file_type(file_path)
        logger.info(f"Processing {file_path} as {file_type}")
        
        all_text = []
        
        if file_type == 'pdf':
            # Analyze PDF content
            analysis = self.analyze_pdf_content(file_path)
            
            # Extract text using appropriate methods
            if analysis['has_text']:
                text_content = self.extract_text_from_pdf(file_path)
                if text_content.strip():
                    all_text.append("=== TEXT CONTENT ===\n" + text_content)
            
            if analysis['has_images']:
                ocr_content = self.extract_images_from_pdf_ocr(file_path, analysis['image_pages'])
                if ocr_content.strip():
                    all_text.append("=== OCR CONTENT ===\n" + ocr_content)
            
            if analysis['has_tables'] and self.textract_client:
                table_content = self.extract_tables_from_pdf_textract(file_path, analysis['table_pages'])
                if table_content.strip():
                    all_text.append("=== TABLE CONTENT ===\n" + table_content)
        
        elif file_type == 'image':
            image_text = self.extract_from_image(file_path)
            if image_text.strip():
                all_text.append(image_text)
        
        elif file_type == 'docx':
            docx_text = self.extract_from_docx(file_path)
            if docx_text.strip():
                all_text.append(docx_text)
        
        elif file_type == 'text':
            text_content = self.extract_from_text_file(file_path)
            if text_content.strip():
                all_text.append(text_content)
        
        else:
            logger.warning(f"Unsupported file type: {file_type}")
            return ""
        
        return "\n\n".join(all_text)

    def save_extracted_text(self, text: str, original_filename: str) -> str:
        """Save extracted text to output folder"""
        output_filename = f"{Path(original_filename).stem}_extracted.txt"
        output_path = os.path.join(self.output_folder, output_filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(text)
            logger.info(f"Saved extracted text to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save text: {e}")
            return ""

    def process_all_files(self):
        """Process all files in the input folder"""
        if not os.path.exists(self.input_folder):
            logger.error(f"Input folder {self.input_folder} does not exist")
            return
        
        files = [f for f in os.listdir(self.input_folder) 
                if os.path.isfile(os.path.join(self.input_folder, f))]
        
        if not files:
            logger.info(f"No files found in {self.input_folder}")
            return
        
        logger.info(f"Found {len(files)} files to process")
        
        for filename in files:
            file_path = os.path.join(self.input_folder, filename)
            logger.info(f"Processing file: {filename}")
            
            try:
                extracted_text = self.process_single_file(file_path)
                
                if extracted_text.strip():
                    output_path = self.save_extracted_text(extracted_text, filename)
                    logger.info(f"Successfully processed {filename} -> {output_path}")
                else:
                    logger.warning(f"No text extracted from {filename}")
            
            except Exception as e:
                logger.error(f"Failed to process {filename}: {e}")

def main():
    """Main function to run the intelligent text extractor"""
    import io  # Add this import for the OCR function
    
    # Initialize the extractor
    extractor = IntelligentTextExtractor()
    
    # Check if Tesseract is available
    try:
        pytesseract.get_tesseract_version()
        logger.info("Tesseract OCR is available")
    except Exception as e:
        logger.error(f"Tesseract OCR not found: {e}")
        logger.error("Please install Tesseract OCR and add it to your PATH")
        return
    
    # Process all files
    extractor.process_all_files()
    
    logger.info("Text extraction completed!")

if __name__ == "__main__":
    main()