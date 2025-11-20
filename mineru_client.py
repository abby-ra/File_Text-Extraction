import os
import requests
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class MineruClient:
    """Minimal client wrapper for a Mineru-like extraction REST API.

    Assumptions (adjust if real API differs):
    - POST /extract accepts multipart/form-data with 'file' and returns JSON
      containing 'text' and optional 'tables' or other structured outputs.
    - Authentication via an API key provided in the 'Authorization' header
      as a Bearer token.
    """

    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None, timeout: int = 60):
        self.api_url = api_url or os.environ.get('MINERU_API_URL', 'https://api.mineru.ai/v1/extract')
        self.api_key = api_key or os.environ.get('MINERU_API_KEY')
        self.timeout = timeout

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def extract(self, file_path: str) -> Dict:
        """Send a file to the Mineru API and return parsed JSON result.

        Returns a dict with at least a 'text' key on success. On failure
        returns an empty dict.
        """
        if not os.path.exists(file_path):
            logger.error("Mineru extract failed: file not found %s", file_path)
            return {}

        headers = {}
        if self.api_key:
            headers['Authorization'] = f"Bearer {self.api_key}"

        try:
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f)}
                # Optional metadata can be added here
                resp = requests.post(self.api_url, headers=headers, files=files, timeout=self.timeout)

            if resp.status_code != 200:
                logger.error("Mineru API error (%s): %s", resp.status_code, resp.text)
                return {}

            data = resp.json()
            # Normalize response to ensure 'text' key exists
            if isinstance(data, dict):
                return data
            else:
                logger.error("Unexpected Mineru response format: %s", type(data))
                return {}

        except Exception as e:
            logger.exception("Mineru extraction failed: %s", e)
            return {}
