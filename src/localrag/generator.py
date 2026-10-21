# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Guilherme Pupini
"""Small synchronous client for Ollama's local chat HTTP API."""

import json
import socket
from typing import TypedDict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_GENERATION_MODEL = "llama3.2:3b"


class ChatMessage(TypedDict):
    role: str
    content: str


class OllamaError(RuntimeError):
    """Base class for local Ollama request and response errors."""


class OllamaUnavailableError(OllamaError):
    """Ollama is not running or cannot be reached."""


class OllamaModelNotFoundError(OllamaError):
    """The requested model is not available in the local Ollama instance."""


class OllamaTimeoutError(OllamaError):
    """The Ollama request exceeded its configured timeout."""


class InvalidOllamaResponseError(OllamaError):
    """Ollama returned a malformed or incomplete chat response."""


class OllamaChatClient:
    """Call the non-streaming ``/api/chat`` endpoint using only stdlib HTTP."""

    def __init__(
        self,
        model: str = DEFAULT_GENERATION_MODEL,
        api_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = 120.0,
    ) -> None:
        if not model or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not api_url or not api_url.strip():
            raise ValueError("api_url must be a non-empty string")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        self.model = model
        self.api_url = api_url
        self.timeout = timeout

    def generate(self, messages: list[ChatMessage]) -> str:
        """Send chat messages and return the assistant's text content."""
        payload = json.dumps(
            {"model": self.model, "messages": messages, "stream": False}
        ).encode("utf-8")
        request = Request(
            self.api_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = response.read()
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            if error.code == 404 and self._looks_like_missing_model(body):
                raise OllamaModelNotFoundError(
                    f"Ollama model '{self.model}' was not found; pull it with `ollama pull {self.model}`"
                ) from error
            raise OllamaUnavailableError(
                f"Ollama returned HTTP {error.code}: {body[:500]}"
            ) from error
        except (socket.timeout, TimeoutError) as error:
            raise OllamaTimeoutError(
                f"Ollama request timed out after {self.timeout:g} seconds"
            ) from error
        except URLError as error:
            if isinstance(error.reason, (socket.timeout, TimeoutError)):
                raise OllamaTimeoutError(
                    f"Ollama request timed out after {self.timeout:g} seconds"
                ) from error
            raise OllamaUnavailableError(
                f"Could not connect to Ollama at {self.api_url}: {error.reason}"
            ) from error

        try:
            response_data = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise InvalidOllamaResponseError("Ollama returned invalid JSON") from error
        if not isinstance(response_data, dict):
            raise InvalidOllamaResponseError("Ollama response must be a JSON object")
        message = response_data.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise InvalidOllamaResponseError(
                "Ollama response is missing message.content text"
            )
        content = message["content"].strip()
        if not content:
            raise InvalidOllamaResponseError("Ollama returned an empty answer")
        return content

    @staticmethod
    def _looks_like_missing_model(body: str) -> bool:
        lowered = body.casefold()
        return "model" in lowered and ("not found" in lowered or "pull" in lowered)
