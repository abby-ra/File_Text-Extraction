"""
DeepDoctection Layout Analyzer
Detects: tables, text blocks, form fields, handwritten regions, images, signatures, stamps, diagrams
"""
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)


class LayoutRegion:
    """Represents a detected region in the document"""
    
    def __init__(self, region_type: str, bbox: tuple, confidence: float, content: Any = None):
        self.region_type = region_type  # table, text, form_field, handwritten, image, signature, stamp, diagram
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.content = content
        self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.region_type,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "content": self.content,
            "metadata": self.metadata
        }


class DeepDoctectionAnalyzer:
    """Layout detection and document structure analysis"""
    
    def __init__(self, config):
        self.config = config
        self.processing_config = config.processing
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize DeepDoctection models"""
        try:
            # Try to import deepdoctection
            import deepdoctection as dd
            
            # Initialize layout detector
            self.layout_detector = dd.get_dd_analyzer(
                config_overwrite=[
                    "PT.LAYOUT.WEIGHTS=microsoft/table-transformer-detection",
                    "PT.ITEM.WEIGHTS=microsoft/table-transformer-structure-recognition"
                ]
            )
            self.deepdoctection_available = True
            logger.info("DeepDoctection initialized successfully")
            
        except ImportError:
            logger.warning("DeepDoctection not available, using fallback detection")
            self.deepdoctection_available = False
            self.layout_detector = None
    
    def analyze_layout(self, image: np.ndarray, page_num: int = 1) -> Dict[str, Any]:
        """
        Analyze document layout and identify regions
        
        Args:
            image: Input image as numpy array
            page_num: Page number for multi-page documents
            
        Returns:
            Dict containing detected regions and metadata
        """
        try:
            if self.deepdoctection_available and self.layout_detector:
                return self._analyze_with_deepdoctection(image, page_num)
            else:
                return self._analyze_with_fallback(image, page_num)
                
        except Exception as e:
            logger.error(f"Layout analysis failed: {e}")
            return {
                "regions": [],
                "page_num": page_num,
                "success": False,
                "error": str(e)
            }
    
    def _analyze_with_deepdoctection(self, image: np.ndarray, page_num: int) -> Dict[str, Any]:
        """Analyze layout using DeepDoctection"""
        import deepdoctection as dd
        from PIL import Image
        
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)
        
        # Run detection
        df = self.layout_detector.analyze(image=pil_image)
        
        regions = []
        
        # Process detected regions
        for page in df:
            for annotation in page.annotations:
                region_type = self._map_annotation_type(annotation.label)
                bbox = (
                    annotation.bbox.x,
                    annotation.bbox.y,
                    annotation.bbox.x + annotation.bbox.width,
                    annotation.bbox.y + annotation.bbox.height
                )
                
                region = LayoutRegion(
                    region_type=region_type,
                    bbox=bbox,
                    confidence=annotation.score if hasattr(annotation, 'score') else 1.0
                )
                
                # Add metadata
                region.metadata["reading_order"] = annotation.reading_order if hasattr(annotation, 'reading_order') else 0
                
                regions.append(region)
        
        # Sort by reading order
        regions.sort(key=lambda r: r.metadata.get("reading_order", 0))
        
        return {
            "regions": [r.to_dict() for r in regions],
            "page_num": page_num,
            "total_regions": len(regions),
            "success": True
        }
    
    def _analyze_with_fallback(self, image: np.ndarray, page_num: int) -> Dict[str, Any]:
        """Fallback layout analysis using traditional CV methods"""
        import cv2
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter small regions
            if w < 20 or h < 20:
                continue
            
            # Classify region based on aspect ratio and size
            aspect_ratio = w / h
            area = w * h
            
            region_type = self._classify_region_fallback(aspect_ratio, area, image[y:y+h, x:x+w])
            
            region = LayoutRegion(
                region_type=region_type,
                bbox=(x, y, x+w, y+h),
                confidence=0.7  # Lower confidence for fallback
            )
            
            regions.append(region)
        
        return {
            "regions": [r.to_dict() for r in regions],
            "page_num": page_num,
            "total_regions": len(regions),
            "success": True,
            "method": "fallback"
        }
    
    def _map_annotation_type(self, label: str) -> str:
        """Map DeepDoctection labels to our region types"""
        mapping = {
            "table": "table",
            "text": "text",
            "title": "text",
            "list": "text",
            "figure": "image",
            "formula": "diagram",
        }
        return mapping.get(label.lower(), "text")
    
    def _classify_region_fallback(self, aspect_ratio: float, area: int, region_img: np.ndarray) -> str:
        """Classify region type using fallback heuristics"""
        import cv2
        
        # Very wide regions are likely text lines
        if aspect_ratio > 5:
            return "text"
        
        # Square-ish regions with significant area might be tables or images
        if 0.5 < aspect_ratio < 2:
            if area > 50000:
                # Check if it has grid structure (table)
                edges = cv2.Canny(region_img, 50, 150)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
                
                if lines is not None and len(lines) > 10:
                    return "table"
                else:
                    return "image"
        
        # Check for handwriting by analyzing stroke patterns
        if self.processing_config.detect_handwriting:
            if self._is_handwritten(region_img):
                return "handwritten"
        
        # Check for signatures (small, contained regions with specific characteristics)
        if self.processing_config.detect_signatures:
            if area < 10000 and aspect_ratio < 3:
                if self._is_signature(region_img):
                    return "signature"
        
        # Check for stamps (circular or rectangular with specific patterns)
        if self.processing_config.detect_stamps:
            if self._is_stamp(region_img):
                return "stamp"
        
        return "text"
    
    def _is_handwritten(self, region_img: np.ndarray) -> bool:
        """Detect if region contains handwriting"""
        import cv2
        
        # Convert to grayscale
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        
        # Calculate stroke width variation
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return False
        
        # Handwriting typically has more irregular contours
        irregularity = np.std([cv2.contourArea(c) for c in contours])
        
        return irregularity > 100
    
    def _is_signature(self, region_img: np.ndarray) -> bool:
        """Detect if region contains a signature"""
        import cv2
        
        # Signatures typically have:
        # 1. Connected components with cursive-like flow
        # 2. Specific aspect ratio
        # 3. Lower pixel density than printed text
        
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Calculate pixel density
        density = np.sum(binary > 0) / binary.size
        
        # Signatures usually have 5-20% pixel density
        return 0.05 < density < 0.20
    
    def _is_stamp(self, region_img: np.ndarray) -> bool:
        """Detect if region contains a stamp"""
        import cv2
        
        # Stamps typically have:
        # 1. Circular or rectangular borders
        # 2. Uniform color (often red or blue)
        # 3. Text arranged in circular pattern
        
        # Detect circles
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=100
        )
        
        return circles is not None
    
    def detect_form_fields(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect form fields (checkboxes, text boxes, radio buttons)
        
        Returns:
            List of detected form fields with their locations
        """
        import cv2
        
        form_fields = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect checkboxes (small squares)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if it's a small square (checkbox)
            if 10 < w < 30 and 10 < h < 30 and abs(w - h) < 5:
                form_fields.append({
                    "type": "checkbox",
                    "bbox": (x, y, x+w, y+h),
                    "checked": self._is_checkbox_checked(binary[y:y+h, x:x+w])
                })
        
        return form_fields
    
    def _is_checkbox_checked(self, checkbox_img: np.ndarray) -> bool:
        """Determine if a checkbox is checked"""
        # Calculate percentage of filled pixels
        filled_ratio = np.sum(checkbox_img > 0) / checkbox_img.size
        
        # Checkbox is checked if more than 30% filled
        return filled_ratio > 0.30
