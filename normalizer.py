"""
Docling Normalizer
Converts all extracted content into canonical structured format (JSON/Markdown)
"""
import logging
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class DoclingNormalizer:
    """Normalize and structure extracted content"""
    
    def __init__(self, config):
        self.config = config
        self.output_config = config.output
    
    def normalize_document(self, 
                          file_path: Path,
                          extracted_data: Dict[str, Any],
                          layout_data: Dict[str, Any],
                          metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert extracted content to canonical structured format
        
        Args:
            file_path: Original document path
            extracted_data: Raw extracted content
            layout_data: Layout detection results
            metadata: Additional metadata
            
        Returns:
            Structured document in canonical format
        """
        try:
            # Create structured document
            structured_doc = {
                "document_info": {
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "file_type": file_path.suffix.lower(),
                    "processing_date": datetime.now().isoformat(),
                    "file_size": file_path.stat().st_size if file_path.exists() else 0
                },
                "content": {
                    "full_text": extracted_data.get("text", ""),
                    "structured_elements": []
                },
                "layout": {
                    "regions": layout_data.get("regions", []),
                    "total_regions": layout_data.get("total_regions", 0)
                },
                "metadata": metadata,
                "processing_info": {
                    "extraction_method": extracted_data.get("method", "unknown"),
                    "confidence": extracted_data.get("confidence", 0.0),
                    "ocr_engine": extracted_data.get("engine", "unknown")
                }
            }
            
            # Add structured elements from layout
            self._add_structured_elements(structured_doc, extracted_data, layout_data)
            
            return structured_doc
            
        except Exception as e:
            logger.error(f"Normalization failed: {e}")
            return {
                "error": str(e),
                "success": False
            }
    
    def _add_structured_elements(self, 
                                 structured_doc: Dict[str, Any],
                                 extracted_data: Dict[str, Any],
                                 layout_data: Dict[str, Any]):
        """Add structured elements from extracted and layout data"""
        
        elements = []
        
        # Process regions from layout detection
        for region in layout_data.get("regions", []):
            element = {
                "type": region.get("type", "unknown"),
                "bbox": region.get("bbox"),
                "confidence": region.get("confidence", 0.0),
                "content": region.get("content", ""),
                "metadata": region.get("metadata", {})
            }
            elements.append(element)
        
        # Add structured content from extraction (for Office docs)
        if "structured_content" in extracted_data:
            content = extracted_data["structured_content"]
            
            # Add paragraphs
            if "paragraphs" in content:
                for para in content["paragraphs"]:
                    elements.append({
                        "type": "paragraph",
                        "content": para.get("text", ""),
                        "style": para.get("style", "Normal")
                    })
            
            # Add tables
            if "tables" in content:
                for table in content["tables"]:
                    elements.append({
                        "type": "table",
                        "index": table.get("index", 0),
                        "data": table.get("data", []),
                        "rows": len(table.get("data", [])),
                        "cols": len(table["data"][0]) if table.get("data") else 0
                    })
            
            # Add slides
            if "slides" in content:
                for slide in content["slides"]:
                    elements.append({
                        "type": "slide",
                        "index": slide.get("index", 0),
                        "text_elements": slide.get("text_elements", []),
                        "tables": slide.get("tables", [])
                    })
            
            # Add sheets
            if "sheets" in content:
                for sheet in content["sheets"]:
                    elements.append({
                        "type": "sheet",
                        "name": sheet.get("name", ""),
                        "data": sheet.get("data", []),
                        "rows": len(sheet.get("data", [])),
                        "cols": len(sheet["data"][0]) if sheet.get("data") else 0
                    })
        
        structured_doc["content"]["structured_elements"] = elements
    
    def to_json(self, structured_doc: Dict[str, Any], output_path: Path) -> bool:
        """Save structured document as JSON"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(structured_doc, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved JSON to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save JSON: {e}")
            return False
    
    def to_markdown(self, structured_doc: Dict[str, Any], output_path: Path) -> bool:
        """Convert structured document to Markdown"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            markdown = self._generate_markdown(structured_doc)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown)
            
            logger.info(f"Saved Markdown to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save Markdown: {e}")
            return False
    
    def _generate_markdown(self, structured_doc: Dict[str, Any]) -> str:
        """Generate Markdown from structured document"""
        lines = []
        
        # Add document header
        doc_info = structured_doc.get("document_info", {})
        lines.append(f"# {doc_info.get('file_name', 'Document')}")
        lines.append("")
        lines.append("## Document Information")
        lines.append(f"- **File Type**: {doc_info.get('file_type', 'unknown')}")
        lines.append(f"- **Processing Date**: {doc_info.get('processing_date', 'unknown')}")
        lines.append(f"- **File Size**: {doc_info.get('file_size', 0)} bytes")
        lines.append("")
        
        # Add processing info
        proc_info = structured_doc.get("processing_info", {})
        lines.append("## Processing Information")
        lines.append(f"- **Method**: {proc_info.get('extraction_method', 'unknown')}")
        lines.append(f"- **OCR Engine**: {proc_info.get('ocr_engine', 'N/A')}")
        lines.append(f"- **Confidence**: {proc_info.get('confidence', 0.0):.2%}")
        lines.append("")
        
        # Add content
        content = structured_doc.get("content", {})
        
        lines.append("## Extracted Content")
        lines.append("")
        
        # Add structured elements
        for element in content.get("structured_elements", []):
            elem_type = element.get("type", "unknown")
            
            if elem_type == "paragraph":
                lines.append(element.get("content", ""))
                lines.append("")
            
            elif elem_type == "table":
                lines.append(f"### Table {element.get('index', 0) + 1}")
                lines.append("")
                
                table_data = element.get("data", [])
                if table_data:
                    # Header row
                    lines.append("| " + " | ".join(table_data[0]) + " |")
                    lines.append("|" + "|".join(["---" for _ in table_data[0]]) + "|")
                    
                    # Data rows
                    for row in table_data[1:]:
                        lines.append("| " + " | ".join(row) + " |")
                    lines.append("")
            
            elif elem_type == "slide":
                lines.append(f"### Slide {element.get('index', 0) + 1}")
                lines.append("")
                for text in element.get("text_elements", []):
                    lines.append(text)
                    lines.append("")
            
            elif elem_type == "sheet":
                lines.append(f"### Sheet: {element.get('name', 'Unknown')}")
                lines.append("")
                
                sheet_data = element.get("data", [])[:10]  # Limit to first 10 rows
                if sheet_data:
                    for row in sheet_data:
                        lines.append("| " + " | ".join(row) + " |")
                    lines.append("")
            
            elif elem_type in ["text", "handwritten"]:
                lines.append(f"**[{elem_type.title()}]**")
                lines.append(element.get("content", ""))
                lines.append("")
            
            elif elem_type == "image":
                bbox = element.get("bbox", [])
                lines.append(f"![Image at {bbox}](image_{bbox})")
                if element.get("content"):
                    lines.append(f"\n*Description: {element['content']}*")
                lines.append("")
        
        # If no structured elements, add full text
        if not content.get("structured_elements"):
            lines.append(content.get("full_text", ""))
        
        # Add layout information
        layout = structured_doc.get("layout", {})
        if layout.get("total_regions", 0) > 0:
            lines.append("## Layout Analysis")
            lines.append(f"- **Total Regions**: {layout['total_regions']}")
            
            region_types = {}
            for region in layout.get("regions", []):
                rtype = region.get("type", "unknown")
                region_types[rtype] = region_types.get(rtype, 0) + 1
            
            lines.append("- **Region Types**:")
            for rtype, count in region_types.items():
                lines.append(f"  - {rtype}: {count}")
            lines.append("")
        
        return "\n".join(lines)
    
    def to_text(self, structured_doc: Dict[str, Any], output_path: Path) -> bool:
        """Save extracted text as plain text file"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get the full text content
            full_text = structured_doc.get("content", {}).get("full_text", "")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            logger.info(f"Saved text to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save text: {e}")
            return False
    
    def save_output(self, structured_doc: Dict[str, Any], base_output_path: Path) -> Dict[str, Path]:
        """
        Save structured document in configured formats
        
        Returns:
            Dict mapping format to output path
        """
        output_paths = {}
        
        output_format = self.output_config.output_format
        
        # Always save plain text file
        txt_path = base_output_path.with_suffix(".txt")
        if self.to_text(structured_doc, txt_path):
            output_paths["text"] = txt_path
        
        if output_format in ["json", "both"]:
            json_path = base_output_path.with_suffix(".json")
            if self.to_json(structured_doc, json_path):
                output_paths["json"] = json_path
        
        if output_format in ["markdown", "both"]:
            md_path = base_output_path.with_suffix(".md")
            if self.to_markdown(structured_doc, md_path):
                output_paths["markdown"] = md_path
        
        return output_paths
