# File_Text-Extraction

This repository extracts text from a variety of input documents, summarizes them, converts summaries to PDF, and can translate between English and Indic languages (example: Malayalam).

Mineru integration
------------------
This project includes an optional integration with a Mineru-like extraction REST API. When configured, the extractor will attempt to send each input file to the remote Mineru endpoint and use the returned structured extraction (text, tables) if available. If Mineru is not configured or fails, the code falls back to the local extraction pipeline (PyMuPDF + Tesseract, DOCX/text readers, optional AWS Textract for tables).

To enable Mineru integration, set environment variables (PowerShell example):

```powershell
$env:MINERU_API_KEY = 'your_api_key_here'
$env:MINERU_API_URL = 'https://api.mineru.ai/v1/extract'  # optional
$env:USE_MINERU = '1'  # optional toggle (client also checks for API key)
```

Notes & assumptions
- The Mineru client (`mineru_client.py`) contains a conservative generic implementation that posts files to an assumed `/extract` endpoint and expects JSON with a `text` field. If your real Mineru API differs, update `mineru_client.py` accordingly.
- The repository still supports the original local extraction pipeline and will use it when Mineru is not available.

See the rest of the README and the code for details on how to run the pipeline.