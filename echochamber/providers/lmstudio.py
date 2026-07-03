"""LM Studio local provider."""

from typing import Optional

from .base import LLMProvider, LLMResponse, Message


def get_lmstudio_model(base_url: str = "http://localhost:1234/v1") -> Optional[str]:
    """
    Query LM Studio for the currently loaded model.

    Returns:
        Model name if available, None otherwise
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key="lm-studio", base_url=base_url)
        models = client.models.list()
        if models.data:
            return models.data[0].id
    except Exception:
        pass
    return None


class LMStudioProvider(LLMProvider):
    """Provider for locally hosted LM Studio models."""

    def __init__(
        self,
        model: str = "local-model",
        base_url: str = "http://localhost:1234/v1",
        auto_detect_model: bool = True,
    ):
        self.base_url = base_url

        try:
            from openai import OpenAI
            # LM Studio uses OpenAI-compatible API
            self.client = OpenAI(
                api_key="lm-studio",  # LM Studio doesn't need a real key
                base_url=base_url,
            )
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        # Auto-detect model if requested and model is default
        if auto_detect_model and model == "local-model":
            detected = get_lmstudio_model(base_url)
            self.model = detected if detected else model
        else:
            self.model = model

    def complete(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        # Convert to OpenAI format
        lm_messages = []

        if system_prompt:
            lm_messages.append({
                "role": "system",
                "content": system_prompt,
            })

        for msg in messages:
            lm_messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=lm_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content,
            model=self.model,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            } if response.usage else None,
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
        )

    @property
    def name(self) -> str:
        return f"lmstudio/{self.model}"
