"""
AWS Bedrock judge for DeepEval metrics.

Usage:
    from bedrock_judge import get_bedrock_judge
    judge = get_bedrock_judge()
    metric = AnswerRelevancyMetric(model=judge)

Requirements:
    pip install boto3 requests
    
    Configure AWS credentials via environment variables:
    
    Option A - IAM credentials (recommended for long-term):
        AWS_ACCESS_KEY_ID=AKIA...
        AWS_SECRET_ACCESS_KEY=...
        AWS_DEFAULT_REGION=eu-north-1
    
    Option B - Temporary credentials (from Bedrock console):
        AWS_ACCESS_KEY_ID=ASIA...
        AWS_SECRET_ACCESS_KEY=...
        AWS_SESSION_TOKEN=...
        AWS_DEFAULT_REGION=eu-north-1
"""
import os
import json
from typing import Optional
from deepeval.models import DeepEvalBaseLLM


class BedrockJudge(DeepEvalBaseLLM):
    """DeepEval-compatible wrapper for AWS Bedrock models using boto3 directly."""

    def __init__(
        self,
        model_id: str = "eu.amazon.nova-lite-v1:0",
        region_name: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.model_id = model_id
        self.region_name = region_name or os.getenv("AWS_DEFAULT_REGION", "eu-north-1")
        self.temperature = temperature
        self._client = None

    def load_model(self):
        if self._client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError(
                    "boto3 is required for Bedrock judge. "
                    "Install with: pip install boto3"
                )
            # boto3 automatically uses AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
            # and AWS_SESSION_TOKEN (if present) from environment
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.region_name,
            )
        return self._client

    def generate(self, prompt: str) -> str:
        client = self.load_model()
        
        # Build request body based on model type
        if "anthropic" in self.model_id or "claude" in self.model_id:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
        elif "titan" in self.model_id:
            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 4096,
                    "temperature": self.temperature,
                },
            }
        elif "meta" in self.model_id or "llama" in self.model_id:
            body = {
                "prompt": prompt,
                "max_gen_len": 4096,
                "temperature": self.temperature,
            }
        elif "nova" in self.model_id or "amazon" in self.model_id:
            body = {
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "inferenceConfig": {
                    "maxTokens": 4096,
                    "temperature": self.temperature,
                },
            }
        else:
            # Generic fallback for Claude-like models
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}],
            }

        response = client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        
        response_body = json.loads(response["body"].read())
        
        # Parse response based on model type
        if "anthropic" in self.model_id or "claude" in self.model_id:
            return response_body.get("content", [{}])[0].get("text", "")
        elif "titan" in self.model_id:
            return response_body.get("results", [{}])[0].get("outputText", "")
        elif "meta" in self.model_id or "llama" in self.model_id:
            return response_body.get("generation", "")
        elif "nova" in self.model_id or "amazon" in self.model_id:
            output = response_body.get("output", {})
            message = output.get("message", {})
            content = message.get("content", [{}])
            return content[0].get("text", "") if content else ""
        else:
            # Try common response formats
            if "content" in response_body:
                return response_body["content"][0].get("text", "")
            return str(response_body)

    async def a_generate(self, prompt: str) -> str:
        # boto3 is synchronous; wrap in thread for async
        import asyncio
        return await asyncio.to_thread(self.generate, prompt)

    def get_model_name(self) -> str:
        return f"AWS Bedrock ({self.model_id})"


def get_bedrock_judge(
    model_id: str = None,
    region_name: str = None,
) -> BedrockJudge:
    """
    Factory to create a Bedrock judge with sensible defaults.

    Args:
        model_id: Bedrock model ID. Defaults to Claude 3 Haiku (fast, cheap).
                  Other options:
                    - anthropic.claude-3-haiku-20240307-v1:0  (fastest/cheapest)
                    - anthropic.claude-3-sonnet-20240229-v1:0  (balanced)
                    - anthropic.claude-3-5-sonnet-20240620-v1:0  (best quality)
                    - amazon.titan-text-premier-v1:0
                    - meta.llama3-8b-instruct-v1:0
        region_name: AWS region. Defaults to AWS_DEFAULT_REGION env var or eu-north-1.
    """
    if model_id is None:
        model_id = os.getenv(
            "DEEPEVAL_BEDROCK_MODEL",
            "eu.amazon.nova-lite-v1:0",
        )
    return BedrockJudge(model_id=model_id, region_name=region_name)
