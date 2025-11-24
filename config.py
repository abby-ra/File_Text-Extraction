"""
Configuration Manager for Hybrid Document Processing Pipeline
"""
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class OCRConfig(BaseModel):
    """OCR configuration settings"""
    primary_ocr: str = Field(default="paddleocr", description="Primary OCR engine")
    tesseract_path: Optional[str] = Field(default=None, description="Tesseract executable path")
    languages: list[str] = Field(default=["en", "ml"], description="OCR languages")
    

class ModelConfig(BaseModel):
    """AI model configuration"""
    trocr_model: str = Field(default="microsoft/trocr-large-handwritten")
    reasoning_model: str = Field(default="gpt-4o")
    use_gpu: bool = Field(default=False)
    

class ProcessingConfig(BaseModel):
    """Processing pipeline configuration"""
    max_workers: int = Field(default=4, ge=1, le=16)
    batch_size: int = Field(default=8, ge=1, le=32)
    enable_preprocessing: bool = Field(default=False)  # Disable for faster processing
    enable_layout_detection: bool = Field(default=True)
    enable_vision_analysis: bool = Field(default=False)  # Vision models removed
    enable_reasoning: bool = Field(default=True)
    
    # Preprocessing options
    deskew_enabled: bool = Field(default=True)
    denoise_enabled: bool = Field(default=True)
    shadow_removal_enabled: bool = Field(default=True)
    page_normalization_enabled: bool = Field(default=True)
    
    # Detection options
    detect_handwriting: bool = Field(default=True)
    detect_signatures: bool = Field(default=True)
    detect_stamps: bool = Field(default=True)


class OutputConfig(BaseModel):
    """Output configuration"""
    output_format: str = Field(default="both", pattern="^(json|markdown|both)$")
    save_intermediate: bool = Field(default=True)
    output_dir: Path = Field(default=Path("output"))


class PipelineConfig(BaseModel):
    """Main pipeline configuration"""
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)
    
    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Create configuration from environment variables"""
        return cls(
            ocr=OCRConfig(
                primary_ocr=os.getenv("PRIMARY_OCR", "paddleocr"),
                tesseract_path=os.getenv("TESSERACT_PATH"),
                languages=os.getenv("LANGUAGES", "en,ml").split(",")
            ),
            models=ModelConfig(
                trocr_model=os.getenv("TROCR_MODEL", "microsoft/trocr-large-handwritten"),
                reasoning_model=os.getenv("REASONING_MODEL", "gpt-4o"),
                use_gpu=os.getenv("USE_GPU", "false").lower() == "true"
            ),
            processing=ProcessingConfig(
                max_workers=int(os.getenv("MAX_WORKERS", "4")),
                batch_size=int(os.getenv("BATCH_SIZE", "8")),
                enable_preprocessing=os.getenv("ENABLE_PREPROCESSING", "true").lower() == "true",
                enable_layout_detection=os.getenv("ENABLE_LAYOUT_DETECTION", "true").lower() == "true",
                enable_vision_analysis=os.getenv("ENABLE_VISION_ANALYSIS", "true").lower() == "true",
                enable_reasoning=os.getenv("ENABLE_REASONING", "true").lower() == "true",
                deskew_enabled=os.getenv("DESKEW_ENABLED", "true").lower() == "true",
                denoise_enabled=os.getenv("DENOISE_ENABLED", "true").lower() == "true",
                shadow_removal_enabled=os.getenv("SHADOW_REMOVAL_ENABLED", "true").lower() == "true",
                page_normalization_enabled=os.getenv("PAGE_NORMALIZATION_ENABLED", "true").lower() == "true",
                detect_handwriting=os.getenv("DETECT_HANDWRITING", "true").lower() == "true",
                detect_signatures=os.getenv("DETECT_SIGNATURES", "true").lower() == "true",
                detect_stamps=os.getenv("DETECT_STAMPS", "true").lower() == "true"
            ),
            output=OutputConfig(
                output_format=os.getenv("OUTPUT_FORMAT", "both"),
                save_intermediate=os.getenv("SAVE_INTERMEDIATE", "true").lower() == "true",
                output_dir=Path(os.getenv("OUTPUT_DIR", "output"))
            ),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    
    def validate_api_keys(self) -> bool:
        """Check if required API keys are present"""
        if self.processing.enable_reasoning:
            if self.models.reasoning_model.startswith("gpt") and not self.openai_api_key:
                return False
            if self.models.reasoning_model.startswith("claude") and not self.anthropic_api_key:
                return False
        return True


# Global configuration instance
config = PipelineConfig.from_env()
