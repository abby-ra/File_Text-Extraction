"""
MinerU Preprocessor Module
Handles preprocessing: deskewing, denoising, shadow removal, page normalization
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class MinerUPreprocessor:
    """Preprocessor for document images to improve OCR accuracy"""
    
    def __init__(self, config):
        self.config = config
        self.processing_config = config.processing
        
    def preprocess_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Apply preprocessing pipeline to image
        
        Returns:
            Dict containing preprocessed image and metadata
        """
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            original_shape = img.shape
            metadata = {
                "original_shape": original_shape,
                "preprocessing_steps": []
            }
            
            # Apply preprocessing steps
            if self.processing_config.denoise_enabled:
                img = self._denoise(img)
                metadata["preprocessing_steps"].append("denoise")
            
            if self.processing_config.shadow_removal_enabled:
                img = self._remove_shadow(img)
                metadata["preprocessing_steps"].append("shadow_removal")
            
            if self.processing_config.deskew_enabled:
                img, angle = self._deskew(img)
                metadata["preprocessing_steps"].append("deskew")
                metadata["deskew_angle"] = angle
            
            if self.processing_config.page_normalization_enabled:
                img = self._normalize_page(img)
                metadata["preprocessing_steps"].append("page_normalization")
            
            metadata["final_shape"] = img.shape
            
            return {
                "image": img,
                "metadata": metadata,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Preprocessing failed for {image_path}: {e}")
            return {
                "image": None,
                "metadata": {"error": str(e)},
                "success": False
            }
    
    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """Apply denoising to reduce image noise"""
        # Convert to grayscale for denoising
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Non-local Means Denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # Convert back to BGR
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    
    def _remove_shadow(self, img: np.ndarray) -> np.ndarray:
        """Remove shadows from document images"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply morphological operations
        dilated_img = cv2.dilate(gray, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        
        # Calculate difference
        diff_img = 255 - cv2.absdiff(gray, bg_img)
        
        # Normalize
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        
        # Convert back to BGR
        return cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)
    
    def _deskew(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        """Detect and correct skew in document images"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is None:
            return img, 0.0
        
        # Calculate average angle
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            angles.append(angle)
        
        median_angle = np.median(angles)
        
        # Rotate image if skew detected
        if abs(median_angle) > 0.5:  # Only correct if skew > 0.5 degrees
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated, median_angle
        
        return img, 0.0
    
    def _normalize_page(self, img: np.ndarray) -> np.ndarray:
        """Normalize page brightness and contrast"""
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        lab = cv2.merge([l, a, b])
        
        # Convert back to BGR
        normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return normalized
    
    def preprocess_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Preprocess PDF by converting to images and processing each page
        
        Returns:
            Dict containing preprocessed pages and metadata
        """
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(pdf_path)
            preprocessed_pages = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                
                # Convert RGBA to BGR if needed
                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                
                # Apply preprocessing
                result = self.preprocess_image(Path(f"page_{page_num}.tmp"))
                result["image"] = img  # Use converted image instead
                
                preprocessed_pages.append({
                    "page_number": page_num + 1,
                    "preprocessed_image": img,
                    "metadata": result.get("metadata", {})
                })
            
            return {
                "pages": preprocessed_pages,
                "total_pages": len(doc),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"PDF preprocessing failed for {pdf_path}: {e}")
            return {
                "pages": [],
                "total_pages": 0,
                "success": False,
                "error": str(e)
            }
