"""
LLM client wrapper
Uses an OpenAI-compatible interface
"""

import json
import re
from typing import Optional, Dict, Any, List

from openai import OpenAI

from ..config import Config


class LLMClient:
    """LLM client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME
        self.timeout = timeout if timeout is not None else Config.LLM_TIMEOUT_SECONDS
        self.max_retries = (
            max_retries if max_retries is not None else Config.LLM_MAX_RETRIES
        )

        if not self.api_key:
            raise ValueError("LLM_API_KEY is not configured")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """
        Send a chat request

        Args:
            messages: Message list
            temperature: Temperature value
            max_tokens: Maximum token count
            response_format: Response format (for example, JSON mode)
            timeout: Per-request timeout in seconds (optional)

        Returns:
            Model response text
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format
        if timeout is not None:
            kwargs["timeout"] = timeout

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        # Some models include <think> blocks in content; strip them out
        content = re.sub(
            r"<(?:think(?:ing)?|reasoning)[^>]*>[\s\S]*?</(?:think(?:ing)?|reasoning)>",
            "",
            content,
        ).strip()
        return content

    def _extract_json_payload(self, response: str) -> str:
        """Extract a JSON object/array from fenced or wrapped model output."""
        cleaned = response.strip()
        cleaned = re.sub(
            r"^```(?:json)?\s*\n?", "", cleaned, flags=re.IGNORECASE
        )
        cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()

        if cleaned[:1] in "[{":
            return cleaned

        decoder = json.JSONDecoder()
        for start, ch in enumerate(cleaned):
            if ch not in '[{':
                continue
            try:
                _, end = decoder.raw_decode(cleaned[start:])
                return cleaned[start:start + end]
            except json.JSONDecodeError:
                continue

        return cleaned

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Send a chat request and return JSON

        Args:
            messages: Message list
            temperature: Temperature value
            max_tokens: Maximum token count
            timeout: Per-request timeout in seconds (optional)

        Returns:
            Parsed JSON object
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            # response_format removed for Codex Responses API compat
        )
        cleaned_response = self._extract_json_payload(response)

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON returned by LLM: {cleaned_response}")

