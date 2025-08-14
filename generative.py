# generative.py
import os
from google import genai
import logging

logger = logging.getLogger(__name__)

def make_client(api_key: str = None):
    """
    Create and return a genai.Client configured for Gemini Developer API.
    If api_key is None, the SDK can read env vars. See docs.
    """
    if api_key:
        client = genai.Client(api_key=api_key)
    else:
        client = genai.Client()  # uses env var if set
    return client

def generate_gemini_text(prompt: str, api_key: str = None, model: str = "gemini-1.5-flash", max_output_tokens: int = 300):
    client = make_client(api_key)
    try:
        # Simple usage: pass contents as a string (SDK returns response with .text)
        response = client.models.generate_content(model=model, contents=prompt)
        # response.text is what docs show
        text = getattr(response, "text", None)
        if text:
            return text
        # some SDK responses return candidates -> content
        if hasattr(response, "candidates") and len(response.candidates) > 0:
            candidate = response.candidates[0]
            return getattr(candidate, "content", getattr(candidate, "text", str(candidate)))
        return str(response)
    except Exception as e:
        logger.exception("Gemini generation failed")
        return f"[Gemini error] {e}"
