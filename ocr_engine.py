"""
Hybrid OCR Engine
Combines PaddleOCR (printed text), TrOCR (handwritten), and Tesseract (fallback)
"""
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class HybridOCREngine:
    """Hybrid OCR system combining multiple engines"""
    
    def __init__(self, config):
        self.config = config
        self.ocr_config = config.ocr
        self.model_config = config.models
        
        # Initialize OCR engines
        self.paddle_ocr = None
        self.trocr_processor = None
        self.trocr_model = None
        self.tesseract_available = False
        
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize all OCR engines"""
        
        # Initialize PaddleOCR
        try:
            from paddleocr import PaddleOCR
            
            # Initialize English OCR (primary)
            self.paddle_ocr = PaddleOCR(
                use_textline_orientation=True,
                lang='en'
            )
            logger.info("PaddleOCR initialized successfully (English)")
            
            # Try Malayalam support (optional)
            try:
                self.paddle_ocr_ml = PaddleOCR(
                    use_textline_orientation=True,
                    lang='ml'
                )
                logger.info("PaddleOCR Malayalam support enabled")
            except Exception as ml_error:
                logger.warning(f"Malayalam OCR not available: {ml_error}")
                self.paddle_ocr_ml = None
                
        except Exception as e:
            logger.warning(f"PaddleOCR initialization failed: {e}")
            self.paddle_ocr = None
            self.paddle_ocr_ml = None
        
        # Initialize TrOCR for handwriting
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            self.trocr_processor = TrOCRProcessor.from_pretrained(self.model_config.trocr_model)
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained(self.model_config.trocr_model)
            
            if self.model_config.use_gpu:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.trocr_model = self.trocr_model.to(device)
            
            logger.info(f"TrOCR initialized successfully: {self.model_config.trocr_model}")
        except Exception as e:
            logger.warning(f"TrOCR initialization failed: {e}")
            self.trocr_processor = None
            self.trocr_model = None
        
        # Check Tesseract availability
        try:
            import pytesseract
            
            if self.ocr_config.tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = self.ocr_config.tesseract_path
            
            # Test Tesseract
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
            logger.info("Tesseract OCR available")
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")
            self.tesseract_available = False
    
    def extract_text(self, image: np.ndarray, region_type: str = "text", language: str = "en") -> Dict[str, Any]:
        """
        Extract text from image using appropriate OCR engine
        
        Args:
            image: Input image as numpy array
            region_type: Type of region (text, handwritten, table, etc.)
            language: Target language (en, ml)
            
        Returns:
            Dict containing extracted text and metadata
        """
        try:
            # Choose OCR engine based on region type
            if region_type == "handwritten" and self.trocr_model:
                return self._extract_with_trocr(image)
            elif language == "ml" and self.paddle_ocr_ml:
                return self._extract_with_paddle(image, lang="ml")
            elif self.paddle_ocr:
                return self._extract_with_paddle(image, lang="en")
            elif self.tesseract_available:
                return self._extract_with_tesseract(image, language)
            else:
                return {
                    "text": "",
                    "confidence": 0.0,
                    "success": False,
                    "error": "No OCR engine available"
                }
                
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "success": False,
                "error": str(e)
            }
    
    def _extract_with_paddle(self, image: np.ndarray, lang: str = "en") -> Dict[str, Any]:
        """Extract text using PaddleOCR"""
        try:
            logger.debug(f"PaddleOCR input image shape: {image.shape}")
            
            ocr_engine = self.paddle_ocr if lang == "en" else self.paddle_ocr_ml
            
            # Run OCR (PaddleOCR 3.x returns OCRResult object)
            result = ocr_engine.ocr(image)
            
            # Extract text and confidence scores from OCRResult
            texts = []
            confidences = []
            
            if result and len(result) > 0:
                ocr_result = result[0]  # Get first page result
                
                # PaddleOCR 3.x OCRResult is dict-like with 'rec_texts' and 'rec_scores' keys
                if 'rec_texts' in ocr_result and 'rec_scores' in ocr_result:
                    texts = [str(text) for text in ocr_result['rec_texts'] if str(text).strip()]
                    confidences = list(ocr_result['rec_scores'][:len(texts)])
                    logger.debug(f"Extracted {len(texts)} text lines from PaddleOCR")
            
            full_text = "\n".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                "text": full_text,
                "confidence": avg_confidence,
                "engine": f"paddleocr_{lang}",
                "lines": len(texts),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "success": False,
                "error": str(e)
            }
    
    def _extract_with_trocr(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract handwritten text using TrOCR"""
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
            
            # Preprocess image
            pixel_values = self.trocr_processor(pil_image, return_tensors="pt").pixel_values
            
            if self.model_config.use_gpu:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                pixel_values = pixel_values.to(device)
            
            # Generate text
            generated_ids = self.trocr_model.generate(pixel_values)
            generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return {
                "text": generated_text,
                "confidence": 0.85,  # TrOCR doesn't provide confidence, use default
                "engine": "trocr",
                "success": True
            }
            
        except Exception as e:
            logger.error(f"TrOCR extraction failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "success": False,
                "error": str(e)
            }
    
    def _extract_with_tesseract(self, image: np.ndarray, language: str = "en") -> Dict[str, Any]:
        """Extract text using Tesseract (fallback)"""
        try:
            import pytesseract
            from PIL import Image
            
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
            
            # Map language codes
            lang_map = {
                "en": "eng",
                "ml": "mal"  # Malayalam
            }
            tesseract_lang = lang_map.get(language, "eng")
            
            # Extract text with confidence
            data = pytesseract.image_to_data(pil_image, lang=tesseract_lang, output_type=pytesseract.Output.DICT)
            
            # Filter out low confidence results
            texts = []
            confidences = []
            
            for i, conf in enumerate(data['conf']):
                if conf > 30:  # Filter confidence < 30
                    text = data['text'][i].strip()
                    if text:
                        texts.append(text)
                        confidences.append(conf / 100.0)  # Normalize to 0-1
            
            full_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                "text": full_text,
                "confidence": avg_confidence,
                "engine": f"tesseract_{tesseract_lang}",
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "success": False,
                "error": str(e)
            }
    
    def extract_table(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract table structure and content
        
        Returns:
            Dict containing table structure and cell contents
        """
        try:
            # Use PaddleOCR for table extraction
            if self.paddle_ocr:
                result = self._extract_with_paddle(image)
                
                # Additional table structure detection would go here
                # For now, return text with table metadata
                return {
                    "table_text": result["text"],
                    "confidence": result["confidence"],
                    "structure": "detected",  # Placeholder for table structure
                    "success": True
                }
            else:
                return {
                    "table_text": "",
                    "confidence": 0.0,
                    "success": False,
                    "error": "No OCR engine available for table extraction"
                }
                
        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
            return {
                "table_text": "",
                "confidence": 0.0,
                "success": False,
                "error": str(e)
            }
    
    def detect_language(self, image: np.ndarray) -> str:
        """
        Detect primary language in image
        
        Returns:
            Language code (en, ml, etc.)
        """
        try:
            # Try OCR with both languages and compare confidence
            result_en = self._extract_with_paddle(image, lang="en")
            result_ml = self._extract_with_paddle(image, lang="ml") if self.paddle_ocr_ml else {"confidence": 0}
            
            if result_ml["confidence"] > result_en["confidence"]:
                return "ml"
            else:
                return "en"
                
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return "en"  # Default to English
