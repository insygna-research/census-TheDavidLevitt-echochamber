"""Google Gemini provider."""

import os
from typing import Optional

from .base import LLMProvider, LLMResponse, Message


class GeminiProvider(LLMProvider):
    """Provider for Google's Gemini models."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY env var or pass api_key."
            )

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
        except ImportError:
            raise ImportError(
                "Please install google-generativeai: pip install google-generativeai"
            )

    def complete(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        # Create model with system instruction
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        model = self.genai.GenerativeModel(
            model_name=self.model,
            generation_config=generation_config,
            system_instruction=system_prompt if system_prompt else None,
        )

        # Convert messages to Gemini format
        # Gemini uses "user" and "model" roles
        gemini_history = []
        for msg in messages:
            role = "model" if msg.role == "assistant" else "user"
            gemini_history.append({
                "role": role,
                "parts": [msg.content],
            })

        # Start chat with history (all but last message)
        if len(gemini_history) > 1:
            chat = model.start_chat(history=gemini_history[:-1])
            # Send last message
            response = chat.send_message(gemini_history[-1]["parts"][0])
        elif gemini_history:
            # Single message
            response = model.generate_content(gemini_history[0]["parts"][0])
        else:
            # No messages - shouldn't happen but handle it
            response = model.generate_content("Hello")

        # Extract usage if available
        usage = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count,
            }

        return LLMResponse(
            content=response.text,
            model=self.model,
            usage=usage,
            raw_response=None,  # Gemini response isn't easily serializable
        )

    @property
    def name(self) -> str:
        return f"gemini/{self.model}"
