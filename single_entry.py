"""Single-entry function for the Hybrid Document Processing Pipeline.

This module exposes a single function `run_full_pipeline` which runs the
entire pipeline for a single file or directory. It uses the existing
`pipeline.HybridDocumentPipeline` and `config` objects.

Usage:
    from single_entry import run_full_pipeline
    run_full_pipeline("input_files", output_dir="output", output_format="both")

The function is intentionally small and re-uses existing components so
that behavior remains consistent with the rest of the codebase.
"""
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from config import config as global_config
from pipeline import HybridDocumentPipeline

logger = logging.getLogger(__name__)


def run_full_pipeline(input_path: str, output_dir: Optional[str] = None, output_format: str = "both", config_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the complete pipeline for a file or directory.

    Args:
        input_path: Path to a file or directory containing files to process.
        output_dir: Optional output directory (overrides config.output.output_dir).
        output_format: One of 'json', 'markdown', 'both'.
        config_override: Optional dict to override configuration fields (applies shallowly).

    Returns:
        A dict with aggregate results or the single-file result.
    """
    # Apply overrides to the global config if provided (shallow)
    cfg = global_config
    if config_override:
        try:
            for k, v in config_override.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        except Exception as e:
            logger.warning("Failed to apply config_override: %s", e)

    # Override output settings if provided
    if output_dir:
        cfg.output.output_dir = Path(output_dir)
    cfg.output.output_format = output_format

    # Initialize pipeline
    pipeline = HybridDocumentPipeline(cfg)

    path = Path(input_path)
    if not path.exists():
        return {"success": False, "error": f"Input path not found: {input_path}"}

    # If file -> process single doc
    if path.is_file():
        result = pipeline.process_document(path)
        return result

    # If directory -> process all supported files and return summary
    supported_patterns = ["*.pdf", "*.docx", "*.doc", "*.xlsx", "*.xls", "*.pptx", "*.ppt", "*.png", "*.jpg", "*.jpeg", "*.tiff", "*.bmp"]
    files = []
    for pattern in supported_patterns:
        files.extend(path.glob(pattern))

    if not files:
        return {"success": False, "error": "No supported files found in directory"}

    results = pipeline.process_batch(files)
    return {"success": True, "results": results}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run full pipeline via single function call")
    parser.add_argument("input_path", type=str, help="Path to input file or directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--format", type=str, choices=["json", "markdown", "both"], default="both")
    args = parser.parse_args()

    out = run_full_pipeline(args.input_path, output_dir=args.output, output_format=args.format)
    if out.get("success"):
        print("Pipeline completed successfully")
    else:
        print(f"Pipeline failed: {out.get('error')}")
