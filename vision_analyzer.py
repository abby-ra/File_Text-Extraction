"""
Qwen-VL Vision Analyzer
Generates detailed descriptions for images, diagrams, and engineering drawings
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class QwenVisionAnalyzer:
    """Vision analysis using Qwen-VL or Qwen2-VL models"""
    
    def __init__(self, config):
        self.config = config
        self.model_config = config.models
        self.model = None
        self.processor = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Qwen-VL model"""
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            
            logger.info(f"Loading Qwen-VL model: {self.model_config.qwen_vl_model}")
            
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(
                self.model_config.qwen_vl_model,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config.qwen_vl_model,
                trust_remote_code=True,
                device_map="auto" if self.model_config.use_gpu else "cpu"
            )
            
            logger.info("Qwen-VL model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Qwen-VL initialization failed: {e}")
            logger.info("Vision analysis will be limited")
            self.model = None
            self.processor = None
    
    def analyze_image(self, image: np.ndarray, context: str = "") -> Dict[str, Any]:
        """
        Generate detailed description of image content
        
        Args:
            image: Input image as numpy array
            context: Additional context about the image
            
        Returns:
            Dict containing description and analysis
        """
        try:
            if self.model is None or self.processor is None:
                return self._fallback_analysis(image)
            
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
            
            # Prepare prompt
            prompt = self._create_analysis_prompt(context)
            
            # Process inputs
            inputs = self.processor(
                text=prompt,
                images=pil_image,
                return_tensors="pt"
            )
            
            if self.model_config.use_gpu:
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate description
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )
            
            # Decode output
            description = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt from output
            if prompt in description:
                description = description.replace(prompt, "").strip()
            
            return {
                "description": description,
                "model": self.model_config.qwen_vl_model,
                "confidence": 0.9,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return {
                "description": "",
                "success": False,
                "error": str(e)
            }
    
    def analyze_diagram(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze engineering drawings or technical diagrams
        
        Returns:
            Dict containing detailed technical description
        """
        prompt_context = """This is a technical diagram or engineering drawing. 
        Please provide a detailed description including:
        1. Type of diagram (flowchart, circuit, mechanical drawing, etc.)
        2. Main components and their relationships
        3. Labels, annotations, and measurements
        4. Technical specifications visible
        5. Overall purpose or function"""
        
        return self.analyze_image(image, context=prompt_context)
    
    def analyze_chart_graph(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze charts, graphs, and data visualizations
        
        Returns:
            Dict containing data insights and trends
        """
        prompt_context = """This is a chart or graph. 
        Please analyze and describe:
        1. Type of visualization (bar chart, line graph, pie chart, etc.)
        2. Axes labels and scales
        3. Data trends and patterns
        4. Key insights and observations
        5. Any notable peaks, valleys, or anomalies"""
        
        return self.analyze_image(image, context=prompt_context)
    
    def analyze_scene(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze general scene or photograph
        
        Returns:
            Dict containing scene description
        """
        prompt_context = """Describe this image in detail, including:
        1. Main subjects and objects
        2. Setting and environment
        3. Actions or activities
        4. Notable details
        5. Text visible in the image"""
        
        return self.analyze_image(image, context=prompt_context)
    
    def _create_analysis_prompt(self, context: str) -> str:
        """Create analysis prompt for Qwen-VL"""
        if context:
            return f"<image>\n{context}\n\nAnalysis:"
        else:
            return "<image>\nProvide a detailed description of this image:\n\nAnalysis:"
    
    def _fallback_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Fallback analysis when Qwen-VL is not available"""
        import cv2
        
        # Basic image analysis
        height, width = image.shape[:2]
        
        # Detect dominant colors
        avg_color = np.mean(image, axis=(0, 1))
        
        # Estimate complexity
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        
        description = f"Image analysis (fallback mode):\n"
        description += f"- Dimensions: {width}x{height} pixels\n"
        description += f"- Dominant colors: RGB({int(avg_color[2])}, {int(avg_color[1])}, {int(avg_color[0])})\n"
        description += f"- Complexity: {'High' if edge_density > 0.1 else 'Medium' if edge_density > 0.05 else 'Low'}\n"
        description += "\nNote: Install Qwen-VL model for detailed content analysis."
        
        return {
            "description": description,
            "model": "fallback",
            "confidence": 0.3,
            "success": True,
            "fallback": True
        }
    
    def extract_text_from_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Use vision model to extract visible text from image
        This complements OCR by providing context and interpretation
        
        Returns:
            Dict containing extracted text and context
        """
        prompt_context = """Extract all visible text from this image.
        Include:
        1. All readable text, labels, and captions
        2. Context about where the text appears
        3. Formatting and layout of the text
        4. Any partially visible or unclear text"""
        
        return self.analyze_image(image, context=prompt_context)
    
    def classify_image_type(self, image: np.ndarray) -> str:
        """
        Classify the type of image
        
        Returns:
            String indicating image type (diagram, chart, photo, document, etc.)
        """
        try:
            result = self.analyze_image(image, context="What type of image is this? (diagram/chart/photo/document/form)")
            
            description = result.get("description", "").lower()
            
            if any(word in description for word in ["diagram", "technical", "engineering", "schematic"]):
                return "diagram"
            elif any(word in description for word in ["chart", "graph", "plot", "visualization"]):
                return "chart"
            elif any(word in description for word in ["document", "text", "page", "form"]):
                return "document"
            elif any(word in description for word in ["photo", "picture", "scene"]):
                return "photo"
            else:
                return "unknown"
                
        except Exception as e:
            logger.error(f"Image classification failed: {e}")
            return "unknown"
