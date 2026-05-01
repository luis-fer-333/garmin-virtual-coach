"""LLM client abstraction supporting Gemini, OpenAI, and AWS Bedrock."""

import json
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified interface for LLM calls."""

    def __init__(self, provider: "str | None" = None) -> None:
        self.provider = provider or settings.llm_provider

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt to the configured LLM and return the response."""
        if self.provider == "gemini":
            return self._call_gemini(system_prompt, user_prompt)
        elif self.provider == "groq":
            return self._call_groq(system_prompt, user_prompt)
        elif self.provider == "openai":
            return self._call_openai(system_prompt, user_prompt)
        elif self.provider == "bedrock":
            return self._call_bedrock(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> str:
        from google import genai

        client = genai.Client(api_key=settings.gemini_api_key)
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=user_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
                max_output_tokens=2000,
            ),
        )
        return response.text

    def _call_groq(self, system_prompt: str, user_prompt: str) -> str:
        from groq import Groq

        client = Groq(api_key=settings.groq_api_key)
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=2000,
        )
        return response.choices[0].message.content

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=2000,
        )
        return response.choices[0].message.content

    def _call_bedrock(self, system_prompt: str, user_prompt: str) -> str:
        import boto3

        client = boto3.client(
            "bedrock-runtime", region_name=settings.aws_region
        )
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": 0.7,
        })
        response = client.invoke_model(
            modelId=settings.bedrock_model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]
