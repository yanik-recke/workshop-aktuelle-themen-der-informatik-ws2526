"""
OpenAI judge for DeepEval metrics.
Uses the real OpenAI API, ignoring OPENAI_BASE_URL (which may point to local LM Studio).
"""
import os
from deepeval.models import DeepEvalBaseLLM


class OpenAIJudge(DeepEvalBaseLLM):
    """DeepEval-compatible wrapper that explicitly uses the real OpenAI API."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.model_name = model
        self.temperature = temperature
        self._client = None

    def load_model(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai is required. Install with: pip install openai")
            
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key or not api_key.startswith("sk-"):
                raise ValueError("Valid OPENAI_API_KEY not found in environment")
            
            # Explicitly use api.openai.com, ignoring OPENAI_BASE_URL
            self._client = OpenAI(
                api_key=api_key,
                base_url="https://api.openai.com/v1",
            )
        return self._client

    def generate(self, prompt: str) -> str:
        client = self.load_model()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    async def a_generate(self, prompt: str) -> str:
        import asyncio
        return await asyncio.to_thread(self.generate, prompt)

    def get_model_name(self) -> str:
        return f"OpenAI ({self.model_name})"


def get_openai_judge(model: str = "gpt-4o-mini") -> OpenAIJudge:
    """Create an OpenAI judge using the real OpenAI API."""
    return OpenAIJudge(model=model)
