"""
Image Description Module using Qwen-VL Vision Language Model
Analyzes and describes image content, objects, text, and context
"""

import os
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QwenImageDescriber:
    """
    Uses Qwen-VL model to generate detailed descriptions of images
    """
    
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct", device=None):
        """
        Initialize the Qwen-VL model
        
        Args:
            model_name: Hugging Face model identifier (default: Qwen2-VL-2B-Instruct)
            device: Device to run model on (cuda/cpu). Auto-detects if None.
        """
        self.model_name = model_name
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing Qwen-VL model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        try:
            # Load model and processor with resume download enabled
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                resume_download=True,  # Resume if download was interrupted
                low_cpu_mem_usage=True  # Reduce memory usage during loading
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.processor = AutoProcessor.from_pretrained(model_name, resume_download=True)
            
            logger.info("Qwen-VL model loaded successfully")
            self.available = True
            
        except Exception as e:
            logger.error(f"Failed to load Qwen-VL model: {str(e)}")
            self.available = False
            self.model = None
            self.processor = None
    
    def describe_image(self, image_path, prompt=None, max_new_tokens=512):
        """
        Generate a detailed description of an image
        
        Args:
            image_path: Path to the image file
            prompt: Custom prompt (default: asks for detailed description)
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            str: Generated description or error message
        """
        if not self.available:
            return "[ERROR] Qwen-VL model not available"
        
        if not os.path.exists(image_path):
            return f"[ERROR] Image file not found: {image_path}"
        
        try:
            # Default prompt for comprehensive image description
            if prompt is None:
                prompt = """Describe this image in detail. Include:
1. Main objects and subjects
2. Text visible in the image (if any)
3. Colors, layout, and composition
4. Context and scene description
5. Any notable details or features"""
            
            # Prepare messages for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # Prepare inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            inputs = inputs.to(self.device)
            
            # Generate description
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False  # Deterministic output
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return output_text.strip()
            
        except Exception as e:
            logger.error(f"Error generating description for {image_path}: {str(e)}")
            return f"[ERROR] Failed to generate description: {str(e)}"
    
    def describe_image_with_question(self, image_path, question, max_new_tokens=256):
        """
        Ask a specific question about an image
        
        Args:
            image_path: Path to the image file
            question: Question to ask about the image
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            str: Answer to the question
        """
        return self.describe_image(image_path, prompt=question, max_new_tokens=max_new_tokens)
    
    def extract_text_from_image_vlm(self, image_path, max_new_tokens=512):
        """
        Extract and transcribe all visible text from an image using VLM
        
        Args:
            image_path: Path to the image file
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            str: Extracted text
        """
        prompt = """Extract and transcribe all text visible in this image. 
Preserve the layout and formatting as much as possible. 
If there is no text, respond with '[NO TEXT FOUND]'."""
        
        return self.describe_image(image_path, prompt=prompt, max_new_tokens=max_new_tokens)
    
    def analyze_document(self, image_path, max_new_tokens=512):
        """
        Analyze a document image and extract structured information
        
        Args:
            image_path: Path to the document image
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            str: Structured analysis of the document
        """
        prompt = """Analyze this document image. Provide:
1. Document type (e.g., invoice, form, certificate, resume)
2. All visible text content
3. Key information fields (dates, names, numbers)
4. Layout and structure
5. Language(s) used"""
        
        return self.describe_image(image_path, prompt=prompt, max_new_tokens=max_new_tokens)


def main():
    """
    Test the Qwen-VL image describer with sample images
    """
    print("=" * 80)
    print("Qwen-VL Image Describer - Test Script")
    print("=" * 80)
    
    # Initialize describer
    describer = QwenImageDescriber()
    
    if not describer.available:
        print("[ERROR] Qwen-VL model failed to initialize")
        return
    
    # Test with images in input_files folder
    test_images = [
        "input_files/beach.png",
        "input_files/text_image.jpg",
        "input_files/applsci-13-09712-g004-550.jpg",
    ]
    
    for img_path in test_images:
        full_path = os.path.join(os.path.dirname(__file__), img_path)
        
        if os.path.exists(full_path):
            print(f"\n{'=' * 80}")
            print(f"Analyzing: {img_path}")
            print(f"{'=' * 80}")
            
            # Generate description
            description = describer.describe_image(full_path)
            print(f"\n[DESCRIPTION]\n{description}")
            
            # Ask specific question
            question = "What emotion is shown in this image?"
            answer = describer.describe_image_with_question(full_path, question)
            print(f"\n[QUESTION] {question}")
            print(f"[ANSWER] {answer}")
            
            print(f"\n{'=' * 80}\n")
        else:
            print(f"[SKIP] Image not found: {full_path}")
    
    print("\n[COMPLETE] Qwen-VL testing finished")


if __name__ == "__main__":
    main()
