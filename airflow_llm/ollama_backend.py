#!/usr/bin/env python3
"""
Ollama Backend for Local Model Inference
Simple, reliable local model serving using Ollama
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class OllamaResponse:
    """Response from Ollama API"""

    text: str
    success: bool
    latency_ms: float
    model_used: str
    error: str | None = None
    tokens_generated: int = 0
    tokens_per_second: float = 0.0


class OllamaBackend:
    """Simple Ollama backend for local model inference"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.available_models = []
        self.session = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def check_connection(self) -> bool:
        """Check if Ollama is running"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            async with self.session.get(
                f"{self.base_url}/api/tags", timeout=5
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.available_models = [
                        model["name"] for model in data.get("models", [])
                    ]
                    logger.info(
                        f"Ollama connected. Available models: {self.available_models}"
                    )
                    return True
                else:
                    logger.warning(f"Ollama responded with status {response.status}")
                    return False
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False

    async def pull_model(self, model_name: str) -> bool:
        """Pull a model if not available"""
        try:
            logger.info(f"Pulling model {model_name}...")

            if not self.session:
                self.session = aiohttp.ClientSession()

            async with self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=300,  # 5 minutes for model download
            ) as response:
                if response.status == 200:
                    # Read streaming response
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line.decode().strip())
                                if data.get("status") == "success":
                                    logger.info(f"Successfully pulled {model_name}")
                                    return True
                                elif "error" in data:
                                    logger.error(
                                        f"Error pulling {model_name}: {data['error']}"
                                    )
                                    return False
                            except json.JSONDecodeError:
                                continue
                    return True
                else:
                    logger.error(
                        f"Failed to pull model {model_name}: {response.status}"
                    )
                    return False
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False

    async def list_models(self) -> list[str]:
        """List available models"""
        if await self.check_connection():
            return self.available_models
        return []

    async def generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        timeout: int = 60,
    ) -> OllamaResponse:
        """Generate text using Ollama"""
        start_time = time.time()

        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            # Check if model is available, try to pull if not
            if model not in self.available_models:
                if not await self.pull_model(model):
                    return OllamaResponse(
                        text="",
                        success=False,
                        latency_ms=(time.time() - start_time) * 1000,
                        model_used=model,
                        error=f"Model {model} not available and could not be pulled",
                    )

            # Generate request
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "top_k": 40,
                },
            }

            async with self.session.post(
                f"{self.base_url}/api/generate", json=payload, timeout=timeout
            ) as response:
                latency_ms = (time.time() - start_time) * 1000

                if response.status == 200:
                    data = await response.json()

                    generated_text = data.get("response", "")
                    tokens_generated = len(generated_text.split())  # Rough estimate
                    tokens_per_second = (
                        tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0
                    )

                    return OllamaResponse(
                        text=generated_text,
                        success=True,
                        latency_ms=latency_ms,
                        model_used=model,
                        tokens_generated=tokens_generated,
                        tokens_per_second=tokens_per_second,
                    )
                else:
                    error_text = await response.text()
                    return OllamaResponse(
                        text="",
                        success=False,
                        latency_ms=latency_ms,
                        model_used=model,
                        error=f"HTTP {response.status}: {error_text}",
                    )

        except asyncio.TimeoutError:
            latency_ms = (time.time() - start_time) * 1000
            return OllamaResponse(
                text="",
                success=False,
                latency_ms=latency_ms,
                model_used=model,
                error="Request timeout",
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return OllamaResponse(
                text="",
                success=False,
                latency_ms=latency_ms,
                model_used=model,
                error=str(e),
            )

    async def batch_generate(
        self, model: str, prompts: list[str], **kwargs
    ) -> list[OllamaResponse]:
        """Generate for multiple prompts"""
        tasks = []
        for prompt in prompts:
            task = self.generate(model, prompt, **kwargs)
            tasks.append(task)

        return await asyncio.gather(*tasks)

    def get_recommended_model(self, task_type: str) -> str:
        """Get recommended model for task type"""
        recommendations = {
            "sql_generation": "codellama:7b",
            "python_code": "codellama:7b",
            "general_reasoning": "llama3:8b",
            "documentation": "phi3:mini",
            "testing": "codellama:7b",
            "data_analysis": "llama3:8b",
            "pipeline_optimization": "phi3:mini",
            "error_debugging": "codellama:7b",
        }

        recommended = recommendations.get(task_type, "phi3:mini")

        # Use available model if recommended isn't available
        if recommended in self.available_models:
            return recommended
        elif "phi3:mini" in self.available_models:
            return "phi3:mini"
        elif "phi3" in self.available_models:
            return "phi3"
        elif self.available_models:
            return self.available_models[0]
        else:
            return "phi3:mini"  # Default to pull


# Global instance
ollama_backend = OllamaBackend()
