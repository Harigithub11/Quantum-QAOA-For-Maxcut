"""
Gemini API Client
Handles API initialization, rate limiting, and error handling
"""

import os
import time
from typing import Optional, Dict, Any
from pathlib import Path


try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Install with: pip install google-generativeai")


class GeminiClient:
    """
    Client for interacting with Google Gemini API.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-pro"):
        """
        Initialize Gemini client.

        Args:
            api_key: Gemini API key (or None to load from .env)
            model_name: Model to use
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed")

        # Load API key
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY')

        if api_key is None:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Configure
        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(model_name)
        self.chat = None

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds

        print(f"Gemini client initialized with model: {model_name}")

    def generate(
        self,
        prompt: str,
        max_retries: int = 3,
        retry_delay: int = 2
    ) -> str:
        """
        Generate response from Gemini.

        Args:
            prompt: Input prompt
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries (seconds)

        Returns:
            Generated text
        """
        # Rate limiting
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                self.last_request_time = time.time()
                return response.text

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Gemini API error (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                else:
                    raise

        return ""

    def start_chat(self) -> None:
        """Start a chat session."""
        self.chat = self.model.start_chat(history=[])

    def send_message(self, message: str) -> str:
        """
        Send message in chat session.

        Args:
            message: Message to send

        Returns:
            Response text
        """
        if self.chat is None:
            self.start_chat()

        response = self.chat.send_message(message)
        return response.text


# Mock client for testing without API key
class MockGeminiClient:
    """Mock Gemini client for testing."""

    def __init__(self, *args, **kwargs):
        print("Using MockGeminiClient (no API key)")

    def generate(self, prompt: str, **kwargs) -> str:
        return f"[MOCK RESPONSE] Generated response for prompt: {prompt[:50]}..."

    def start_chat(self):
        pass

    def send_message(self, message: str) -> str:
        return f"[MOCK RESPONSE] Response to: {message[:50]}..."


def create_client(api_key: Optional[str] = None, use_mock: bool = False) -> GeminiClient:
    """
    Factory function to create Gemini client.

    Args:
        api_key: API key
        use_mock: Use mock client for testing

    Returns:
        GeminiClient or MockGeminiClient
    """
    if use_mock or not GEMINI_AVAILABLE:
        return MockGeminiClient()

    try:
        return GeminiClient(api_key=api_key)
    except (ValueError, ImportError) as e:
        print(f"Warning: {e}")
        print("Using mock client instead")
        return MockGeminiClient()


if __name__ == "__main__":
    print("Testing Gemini client...")

    # Test with mock
    client = create_client(use_mock=True)
    response = client.generate("Hello, Gemini!")
    print(f"Mock response: {response}")

    print("\nGemini client module ready!")
