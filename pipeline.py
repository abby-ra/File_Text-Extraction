"""
Hybrid Document Processing Pipeline
Main orchestrator coordinating all processing stages
"""
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import cv2
from datetime import datetime

from config import config
from preprocessor import MinerUPreprocessor
from layout_analyzer import DeepDoctectionAnalyzer
from ocr_engine import HybridOCREngine
from office_processor import OfficeDocumentProcessor
from normalizer import DoclingNormalizer
from reasoning_engine import ReasoningEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HybridDocumentPipeline:
    """Complete hybrid document processing pipeline"""
    
    def __init__(self, config_override=None):
        """
        Initialize pipeline with all components
        
        Args:
            config_override: Optional configuration override
        """
        self.config = config_override if config_override else config
        
        logger.info("Initializing Hybrid Document Processing Pipeline")
        
        # Initialize components
        self.preprocessor = MinerUPreprocessor(self.config)
        self.layout_analyzer = DeepDoctectionAnalyzer(self.config)
        self.ocr_engine = HybridOCREngine(self.config)
        self.office_processor = OfficeDocumentProcessor(self.config)
        self.normalizer = DoclingNormalizer(self.config)
        
        # Vision analysis disabled
        self.vision_analyzer = None
        
        if self.config.processing.enable_reasoning:
            self.reasoning_engine = ReasoningEngine(self.config)
        else:
            self.reasoning_engine = None
        
        logger.info("Pipeline initialization complete")
    
    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single document through the complete pipeline
        
        Args:
            file_path: Path to input document
            
        Returns:
            Dict containing processing results
        """
        logger.info(f"Processing document: {file_path.name}")
        
        start_time = datetime.now()
        
        try:
            # Determine document type
            file_type = self._classify_file_type(file_path)
            logger.info(f"Document type: {file_type}")
            
            # Route to appropriate processor
            if file_type == "office":
                result = self._process_office_document(file_path)
            elif file_type == "pdf":
                result = self._process_pdf(file_path)
            elif file_type == "image":
                result = self._process_image(file_path)
            else:
                result = {
                    "success": False,
                    "error": f"Unsupported file type: {file_path.suffix}"
                }
            
            # Add processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            result["processing_time"] = processing_time
            
            logger.info(f"Processing complete in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": str(file_path)
            }
    
    def _classify_file_type(self, file_path: Path) -> str:
        """Classify document type based on extension"""
        suffix = file_path.suffix.lower()
        
        if suffix in ['.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt']:
            return "office"
        elif suffix == '.pdf':
            return "pdf"
        elif suffix in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            return "image"
        else:
            return "unknown"
    
    def _process_office_document(self, file_path: Path) -> Dict[str, Any]:
        """Process Office documents (DOCX, XLSX, PPTX)"""
        logger.info("Processing Office document")
        
        # Extract content
        extracted_data = self.office_processor.process_document(file_path)
        
        if not extracted_data.get("success"):
            return extracted_data
        
        # No layout detection or OCR needed for Office docs
        layout_data = {"regions": [], "total_regions": 0}
        
        # Vision analysis disabled
        
        # Normalize to structured format
        structured_doc = self.normalizer.normalize_document(
            file_path=file_path,
            extracted_data=extracted_data,
            layout_data=layout_data,
            metadata={"document_type": "office"}
        )
        
        # Apply reasoning
        if self.reasoning_engine:
            analysis = self.reasoning_engine.analyze_document(structured_doc)
            structured_doc["analysis"] = analysis
        
        # Save outputs
        output_base = self.config.output.output_dir / file_path.stem
        output_paths = self.normalizer.save_output(structured_doc, output_base)
        
        return {
            "success": True,
            "structured_document": structured_doc,
            "output_paths": output_paths,
            "file_path": str(file_path)
        }
    
    def _process_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Process PDF documents"""
        logger.info("Processing PDF document")
        
        # Preprocess PDF pages
        if self.config.processing.enable_preprocessing:
            preprocess_result = self.preprocessor.preprocess_pdf(file_path)
            
            if not preprocess_result.get("success"):
                return preprocess_result
            
            pages = preprocess_result.get("pages", [])
        else:
            # Load PDF without preprocessing
            import fitz
            doc = fitz.open(file_path)
            pages = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                
                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                
                pages.append({"page_number": page_num + 1, "preprocessed_image": img})
        
        # Process each page
        all_text = []
        all_regions = []
        
        for page_data in pages:
            page_num = page_data["page_number"]
            image = page_data["preprocessed_image"]
            
            logger.info(f"Processing page {page_num}")
            
            # Detect layout
            layout_result = self.layout_analyzer.analyze_layout(image, page_num)
            all_regions.extend(layout_result.get("regions", []))
            
            # Extract text from each region
            for region in layout_result.get("regions", []):
                region_type = region["type"]
                bbox = region["bbox"]
                
                # Extract region image
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                region_img = image[y1:y2, x1:x2]
                
                # Apply appropriate extraction method
                if region_type in ["image", "diagram"]:
                    # Vision analysis disabled - mark region only
                    region["content"] = f"[{region_type.upper()}]"
                
                elif region_type == "table":
                    # Extract table
                    table_result = self.ocr_engine.extract_table(region_img)
                    region["content"] = table_result.get("table_text", "")
                    all_text.append(region["content"])
                
                elif region_type in ["text", "handwritten"]:
                    # Extract text using appropriate OCR
                    language = self.ocr_engine.detect_language(region_img)
                    ocr_result = self.ocr_engine.extract_text(region_img, region_type, language)
                    region["content"] = ocr_result.get("text", "")
                    all_text.append(region["content"])
        
        # Combine all extracted content
        full_text = "\n\n".join(all_text)
        
        extracted_data = {
            "text": full_text,
            "success": True,
            "method": "hybrid_pipeline",
            "total_pages": len(pages)
        }
        
        layout_data = {
            "regions": all_regions,
            "total_regions": len(all_regions)
        }
        
        # Normalize
        structured_doc = self.normalizer.normalize_document(
            file_path=file_path,
            extracted_data=extracted_data,
            layout_data=layout_data,
            metadata={"document_type": "pdf", "pages": len(pages)}
        )
        
        # Apply reasoning
        if self.reasoning_engine:
            analysis = self.reasoning_engine.analyze_document(structured_doc)
            structured_doc["analysis"] = analysis
        
        # Save outputs
        output_base = self.config.output.output_dir / file_path.stem
        output_paths = self.normalizer.save_output(structured_doc, output_base)
        
        return {
            "success": True,
            "structured_document": structured_doc,
            "output_paths": output_paths,
            "file_path": str(file_path)
        }
    
    def _process_image(self, file_path: Path) -> Dict[str, Any]:
        """Process image files"""
        logger.info("Processing image file")
        
        # Load image
        image = cv2.imread(str(file_path))
        
        if image is None:
            return {
                "success": False,
                "error": f"Failed to load image: {file_path}"
            }
        
        # Preprocess
        if self.config.processing.enable_preprocessing:
            preprocess_result = self.preprocessor.preprocess_image(file_path)
            
            if preprocess_result.get("success"):
                image = preprocess_result["image"]
        
        # Detect layout
        layout_result = self.layout_analyzer.analyze_layout(image)
        
        # Extract text from regions
        all_text = []
        
        for region in layout_result.get("regions", []):
            region_type = region["type"]
            bbox = region["bbox"]
            
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            region_img = image[y1:y2, x1:x2]
            
            if region_type in ["text", "handwritten"]:
                language = self.ocr_engine.detect_language(region_img)
                ocr_result = self.ocr_engine.extract_text(region_img, region_type, language)
                region["content"] = ocr_result.get("text", "")
                all_text.append(region["content"])
        
        full_text = "\n\n".join(all_text)
        
        extracted_data = {
            "text": full_text,
            "success": True,
            "method": "hybrid_pipeline"
        }
        
        # Normalize
        structured_doc = self.normalizer.normalize_document(
            file_path=file_path,
            extracted_data=extracted_data,
            layout_data=layout_result,
            metadata={"document_type": "image"}
        )
        
        # Apply reasoning
        if self.reasoning_engine:
            analysis = self.reasoning_engine.analyze_document(structured_doc)
            structured_doc["analysis"] = analysis
        
        # Save outputs
        output_base = self.config.output.output_dir / file_path.stem
        output_paths = self.normalizer.save_output(structured_doc, output_base)
        
        return {
            "success": True,
            "structured_document": structured_doc,
            "output_paths": output_paths,
            "file_path": str(file_path)
        }
    
    def process_batch(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """
        Process multiple documents
        
        Args:
            file_paths: List of document paths to process
            
        Returns:
            List of processing results
        """
        results = []
        
        logger.info(f"Processing batch of {len(file_paths)} documents")
        
        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"Processing {i}/{len(file_paths)}: {file_path.name}")
            result = self.process_document(file_path)
            results.append(result)
        
        logger.info(f"Batch processing complete: {len(results)} documents processed")
        
        return results


def main():
    """Main entry point for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid Document Processing Pipeline")
    parser.add_argument("input_path", type=str, help="Path to input file or directory")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--format", type=str, choices=["json", "markdown", "both"], default="both", help="Output format")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    # Update config with CLI arguments
    config.output.output_dir = Path(args.output)
    config.output.output_format = args.format
    
    # Initialize pipeline
    pipeline = HybridDocumentPipeline(config)
    
    # Process input
    if input_path.is_file():
        result = pipeline.process_document(input_path)
        
        if result.get("success"):
            print(f"[SUCCESS] Successfully processed: {input_path.name}")
            print(f"Output: {result.get('output_paths', {})}")
        else:
            print(f"[ERROR] Processing failed: {result.get('error', 'Unknown error')}")
    
    elif input_path.is_dir():
        # Process all supported files in directory
        file_patterns = ["*.pdf", "*.docx", "*.doc", "*.xlsx", "*.xls", "*.pptx", "*.ppt", 
                        "*.png", "*.jpg", "*.jpeg", "*.tiff", "*.bmp"]
        
        files = []
        for pattern in file_patterns:
            files.extend(input_path.glob(pattern))
        
        if not files:
            print("[ERROR] No supported files found in directory")
            return
        
        print(f"📁 Found {len(files)} documents to process")
        
        results = pipeline.process_batch(files)
        
        successful = sum(1 for r in results if r.get("success"))
        print(f"\n[SUCCESS] Successfully processed: {successful}/{len(results)} documents")
    
    else:
        print(f"[ERROR] Invalid input path: {input_path}")


if __name__ == "__main__":
    main()
