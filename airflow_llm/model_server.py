#!/usr/bin/env python3
"""
Production-Grade LLM Model Server with vLLM Backend
High-performance multi-model inference with advanced optimizations
"""

import asyncio
import gc
import logging
import time
from dataclasses import dataclass
from typing import Any

try:
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    vllm = None
    LLM = None
    SamplingParams = None
    AsyncLLMEngine = None
    EngineArgs = None

try:
    import torch
    import torch.cuda

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import accelerate
    from accelerate import Accelerator
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModelForCausalLM = None
    accelerate = None
    Accelerator = None

try:
    import bitsandbytes as bnb

    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    bnb = None

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Advanced configuration for individual model"""

    name: str
    model_path: str
    specialization: str  # sql, python, general, reasoning, etc.
    max_tokens: int = 4096
    temperature: float = 0.1
    quantization: str = "AWQ"  # AWQ, GPTQ, 4bit, 8bit, or None
    gpu_memory_utilization: float = 0.85
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_batch_size: int = 256
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    cpu_offload: bool = False
    load_format: str = "auto"
    dtype: str = "float16"
    kv_cache_dtype: str = "auto"
    trust_remote_code: bool = False
    revision: str | None = None
    tokenizer_revision: str | None = None
    seed: int = 42
    max_parallel_loading_workers: int | None = None


@dataclass
class InferenceResult:
    """Enhanced result from model inference"""

    text: str
    success: bool
    latency_ms: float
    tokens_generated: int
    model_used: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    error: str | None = None
    cached: bool = False
    finish_reason: str | None = None
    logprobs: list[dict] | None = None


@dataclass
class BatchInferenceResult:
    """Result from batch inference"""

    results: list[InferenceResult]
    total_latency_ms: float
    throughput_tokens_per_second: float
    success_rate: float


class AdvancedModelServer:
    """
    Production-grade multi-model server with advanced optimizations:
    - vLLM backend with tensor parallelism
    - Quantization support (AWQ, GPTQ, 4-bit, 8-bit)
    - Automatic batching and caching
    - Dynamic model loading/unloading
    - GPU memory optimization
    - Performance monitoring
    """

    def __init__(self, enable_metrics: bool = True):
        self.models: dict[str, AsyncLLMEngine] = {}
        self.model_configs: dict[str, ModelConfig] = {}
        self.tokenizers: dict[str, Any] = {}
        self.is_initialized = False
        self.enable_metrics = enable_metrics
        self.performance_metrics = {}
        self.cache = {}
        self.batch_queue = asyncio.Queue()
        self.batch_processor_task = None

        # Enhanced model configurations with production optimizations
        self.default_configs = {
            "codellama-7b": ModelConfig(
                name="codellama-7b",
                model_path="codellama/CodeLlama-7b-Instruct-hf",
                specialization="python",
                temperature=0.1,
                quantization="AWQ",
                max_tokens=8192,
                gpu_memory_utilization=0.85,
                enable_chunked_prefill=True,
                enable_prefix_caching=True,
                max_batch_size=128,
            ),
            "sqlcoder-7b": ModelConfig(
                name="sqlcoder-7b",
                model_path="defog/sqlcoder-7b-2",
                specialization="sql",
                temperature=0.0,
                quantization="AWQ",
                max_tokens=4096,
                gpu_memory_utilization=0.80,
                enable_chunked_prefill=True,
                enable_prefix_caching=True,
                max_batch_size=64,
            ),
            "llama3-8b": ModelConfig(
                name="llama3-8b",
                model_path="meta-llama/Meta-Llama-3-8B-Instruct",
                specialization="general",
                temperature=0.2,
                quantization="AWQ",
                max_tokens=8192,
                gpu_memory_utilization=0.85,
                tensor_parallel_size=1,
                enable_chunked_prefill=True,
                enable_prefix_caching=True,
                max_batch_size=256,
            ),
            "phi3-mini": ModelConfig(
                name="phi3-mini",
                model_path="microsoft/Phi-3-mini-4k-instruct",
                specialization="reasoning",
                temperature=0.1,
                quantization="4bit",  # Phi3 works well with 4-bit
                max_tokens=4096,
                gpu_memory_utilization=0.70,
                enable_chunked_prefill=True,
                enable_prefix_caching=True,
                max_batch_size=128,
            ),
            "wizardcoder-7b": ModelConfig(
                name="wizardcoder-7b",
                model_path="WizardLM/WizardCoder-Python-7B-V1.0",
                specialization="python",
                temperature=0.1,
                quantization="AWQ",
                max_tokens=8192,
                gpu_memory_utilization=0.85,
                enable_chunked_prefill=True,
                enable_prefix_caching=True,
                max_batch_size=128,
            ),
            "deepseek-coder-7b": ModelConfig(
                name="deepseek-coder-7b",
                model_path="deepseek-ai/deepseek-coder-7b-instruct-v1.5",
                specialization="python",
                temperature=0.1,
                quantization="AWQ",
                max_tokens=8192,
                gpu_memory_utilization=0.85,
                enable_chunked_prefill=True,
                enable_prefix_caching=True,
                max_batch_size=128,
            ),
        }

        # GPU monitoring
        self.gpu_stats = {
            "memory_used": 0,
            "memory_total": 0,
            "utilization": 0,
            "temperature": 0,
        }

    async def initialize_models(
        self,
        model_names: list[str],
        force_download: bool = False,
        backend: str = "auto",
    ):
        """
        Initialize specified models with backend detection
        """
        # Try different backends in order of preference
        if backend == "auto":
            if VLLM_AVAILABLE and TORCH_AVAILABLE:
                backend = "vllm"
            else:
                backend = "ollama"

        if backend == "vllm":
            return await self._initialize_vllm_models(model_names, force_download)
        elif backend == "ollama":
            return await self._initialize_ollama_models(model_names)
        else:
            raise RuntimeError(f"Unknown backend: {backend}")

    async def _initialize_ollama_models(self, model_names: list[str]):
        """Initialize models using Ollama backend"""
        from .ollama_backend import ollama_backend

        logger.info(f"Initializing models using Ollama backend...")

        async with ollama_backend as backend:
            # Check Ollama connection
            if not await backend.check_connection():
                logger.error(
                    "Ollama not running. Install and start Ollama: https://ollama.ai"
                )
                raise RuntimeError("Ollama is required when vLLM is not available")

            # Get available models
            available_models = await backend.list_models()
            logger.info(f"Available Ollama models: {available_models}")

            # Initialize models
            successful_models = []
            for model_name in model_names:
                try:
                    # Map our model names to Ollama model names
                    ollama_model = self._map_to_ollama_model(model_name)

                    # Test the model
                    test_result = await backend.generate(
                        ollama_model, "Hello, this is a test.", max_tokens=10
                    )

                    if test_result.success:
                        self.models[model_name] = ollama_model
                        self.model_configs[model_name] = self.default_configs.get(
                            model_name
                        )
                        successful_models.append(model_name)
                        logger.info(
                            f"Successfully loaded {model_name} -> {ollama_model}"
                        )
                    else:
                        logger.error(
                            f"Failed to test model {model_name}: {test_result.error}"
                        )

                except Exception as e:
                    logger.error(f"Failed to initialize {model_name}: {e}")
                    continue

            if not successful_models:
                # Try to pull phi3:mini as fallback
                logger.info("No models available, pulling phi3:mini as fallback...")
                if await backend.pull_model("phi3:mini"):
                    self.models["phi3-mini"] = "phi3:mini"
                    self.model_configs["phi3-mini"] = self.default_configs.get(
                        "phi3-mini"
                    )
                    successful_models.append("phi3-mini")
                else:
                    raise RuntimeError(
                        "No models could be loaded and phi3:mini could not be pulled"
                    )

            self.is_initialized = True
            self.backend_type = "ollama"
            logger.info(
                f"Ollama backend initialized with {len(successful_models)} models: {successful_models}"
            )

    def _map_to_ollama_model(self, model_name: str) -> str:
        """Map our model names to Ollama model names"""
        mapping = {
            "phi3-mini": "phi3:mini",
            "codellama-7b": "codellama:7b",
            "llama3-8b": "llama3:8b",
            "sqlcoder-7b": "codellama:7b",  # Use codellama for SQL
            "wizardcoder-7b": "codellama:7b",
            "deepseek-coder-7b": "codellama:7b",
        }
        return mapping.get(model_name, "phi3:mini")

    async def _initialize_vllm_models(
        self, model_names: list[str], force_download: bool = False
    ):
        """Initialize models using vLLM backend (original implementation)"""
        if not VLLM_AVAILABLE:
            logger.error("vLLM not available. Install with: pip install vllm")
            raise RuntimeError("vLLM is required for production model serving")

        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available. Install with: pip install torch")
            raise RuntimeError("PyTorch is required for model serving")

        logger.info(
            f"Initializing {len(model_names)} models with vLLM and advanced optimizations..."
        )

        # Check GPU availability
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"Found {gpu_count} GPU(s) available")
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                logger.info(
                    f"GPU {i}: {props.name}, Memory: {props.total_memory // 1024**3}GB"
                )
        else:
            logger.warning(
                "No GPU available, falling back to CPU (not recommended for production)"
            )

        # Initialize models sequentially to avoid OOM
        successful_models = []

        for model_name in model_names:
            if model_name not in self.default_configs:
                logger.warning(f"Unknown model {model_name}, skipping...")
                continue

            config = self.default_configs[model_name]

            try:
                logger.info(
                    f"Loading {model_name} with {config.specialization} specialization..."
                )
                logger.info(
                    f"Config: quantization={config.quantization}, "
                    f"max_tokens={config.max_tokens}, "
                    f"batch_size={config.max_batch_size}"
                )

                # Prepare engine arguments with advanced optimizations
                engine_args = EngineArgs(
                    model=config.model_path,
                    tokenizer=config.model_path,
                    tensor_parallel_size=config.tensor_parallel_size,
                    pipeline_parallel_size=config.pipeline_parallel_size,
                    gpu_memory_utilization=config.gpu_memory_utilization,
                    max_model_len=config.max_tokens,
                    quantization=config.quantization
                    if config.quantization != "None"
                    else None,
                    dtype=config.dtype,
                    kv_cache_dtype=config.kv_cache_dtype,
                    load_format=config.load_format,
                    trust_remote_code=config.trust_remote_code,
                    revision=config.revision,
                    tokenizer_revision=config.tokenizer_revision,
                    seed=config.seed,
                    max_parallel_loading_workers=config.max_parallel_loading_workers,
                    disable_log_stats=False,
                    enforce_eager=False,
                    disable_custom_all_reduce=False,
                    enable_chunked_prefill=config.enable_chunked_prefill,
                    enable_prefix_caching=config.enable_prefix_caching,
                    cpu_offload_gb=0,
                    max_num_batched_tokens=config.max_batch_size
                    * config.max_tokens
                    // 4,
                    max_num_seqs=config.max_batch_size,
                    preemption_mode="recompute",
                )

                # Initialize async engine
                engine = AsyncLLMEngine.from_engine_args(engine_args)

                # Test the model with a simple prompt
                test_result = await self._test_model(engine, model_name)
                if not test_result:
                    logger.error(f"Model {model_name} failed initialization test")
                    continue

                self.models[model_name] = engine
                self.model_configs[model_name] = config

                # Initialize tokenizer for token counting
                if TRANSFORMERS_AVAILABLE:
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
                        self.tokenizers[model_name] = tokenizer
                    except Exception as e:
                        logger.warning(
                            f"Could not load tokenizer for {model_name}: {e}"
                        )

                # Initialize performance metrics
                if self.enable_metrics:
                    self.performance_metrics[model_name] = {
                        "total_requests": 0,
                        "total_latency": 0,
                        "total_tokens": 0,
                        "avg_latency": 0,
                        "tokens_per_second": 0,
                        "success_rate": 0,
                        "cache_hits": 0,
                        "errors": 0,
                    }

                successful_models.append(model_name)
                logger.info(f"Successfully loaded {model_name}")

                # Update GPU stats
                if torch.cuda.is_available():
                    self._update_gpu_stats()

            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
                # Cleanup any partially loaded model
                if model_name in self.models:
                    del self.models[model_name]
                if model_name in self.model_configs:
                    del self.model_configs[model_name]
                continue

        if not successful_models:
            raise RuntimeError("No models could be loaded successfully")

        self.is_initialized = True

        # Start batch processor
        if self.batch_processor_task is None:
            self.batch_processor_task = asyncio.create_task(self._batch_processor())

        logger.info(
            f"Model server initialized successfully with {len(successful_models)} models: {successful_models}"
        )

        # Log memory usage
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                logger.info(
                    f"GPU {i} memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved"
                )

    async def _test_model(self, engine: AsyncLLMEngine, model_name: str) -> bool:
        """Test model with a simple prompt"""
        try:
            sampling_params = SamplingParams(temperature=0.1, max_tokens=50, top_p=0.9)

            test_prompt = "Hello, how are you?"

            outputs = await engine.generate(
                test_prompt,
                sampling_params,
                request_id=f"test_{model_name}_{int(time.time())}",
            )

            return bool(outputs and outputs[0].outputs)

        except Exception as e:
            logger.error(f"Model test failed for {model_name}: {e}")
            return False

    async def generate(
        self,
        prompt: str,
        model_name: str | None = None,
        max_tokens: int = 1024,
        temperature: float = None,
        top_p: float = 0.9,
        frequency_penalty: float = 0.1,
        presence_penalty: float = 0.1,
        stop: list[str] | None = None,
        stream: bool = False,
        use_cache: bool = True,
        **kwargs,
    ) -> InferenceResult:
        """
        Generate text with advanced sampling parameters
        """
        if not self.is_initialized:
            return InferenceResult(
                text="",
                success=False,
                latency_ms=0,
                tokens_generated=0,
                model_used="none",
                error="Model server not initialized",
            )

        # Auto-select model if not specified
        if model_name is None:
            model_name = self._auto_select_model(prompt)

        if model_name not in self.models:
            return InferenceResult(
                text="",
                success=False,
                latency_ms=0,
                tokens_generated=0,
                model_used=model_name,
                error=f"Model {model_name} not available",
            )

        # Route to appropriate backend
        if hasattr(self, "backend_type") and self.backend_type == "ollama":
            return await self._generate_ollama(
                prompt, model_name, max_tokens, temperature, **kwargs
            )
        else:
            return await self._generate_vllm(
                prompt,
                model_name,
                max_tokens,
                temperature,
                top_p,
                frequency_penalty,
                presence_penalty,
                stop,
                stream,
                use_cache,
                **kwargs,
            )

    async def _generate_ollama(
        self,
        prompt: str,
        model_name: str,
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> InferenceResult:
        """Generate using Ollama backend"""
        from .ollama_backend import ollama_backend

        # Get the Ollama model name
        ollama_model = self.models[model_name]

        # Use default temperature if not provided
        if temperature is None:
            config = self.model_configs.get(model_name)
            temperature = config.temperature if config else 0.7

        async with ollama_backend as backend:
            result = await backend.generate(
                ollama_model, prompt, max_tokens=max_tokens, temperature=temperature
            )

            # Convert to our standard format
            return InferenceResult(
                text=result.text,
                success=result.success,
                latency_ms=result.latency_ms,
                tokens_generated=result.tokens_generated,
                model_used=model_name,
                prompt_tokens=0,  # Ollama doesn't provide this
                completion_tokens=result.tokens_generated,
                total_tokens=result.tokens_generated,
                tokens_per_second=result.tokens_per_second,
                error=result.error,
                cached=False,
            )

    async def _generate_vllm(
        self,
        prompt: str,
        model_name: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        stop: list[str] | None,
        stream: bool,
        use_cache: bool,
        **kwargs,
    ) -> InferenceResult:
        """Generate using vLLM backend (original implementation)"""
        # Check cache first
        cache_key = None
        if use_cache:
            cache_key = self._get_cache_key(prompt, model_name, max_tokens, temperature)
            if cache_key in self.cache:
                cached_result = self.cache[cache_key]
                cached_result.cached = True
                if self.enable_metrics:
                    self.performance_metrics[model_name]["cache_hits"] += 1
                return cached_result

        config = self.model_configs[model_name]
        engine = self.models[model_name]

        # Use config defaults if not overridden
        if temperature is None:
            temperature = config.temperature

        if stop is None:
            stop = ["<|endoftext|>", "<|end|>", "</s>", "<|im_end|>"]

        start_time = time.time()

        try:
            # Count prompt tokens
            prompt_tokens = 0
            if model_name in self.tokenizers:
                try:
                    prompt_tokens = len(self.tokenizers[model_name].encode(prompt))
                except:
                    prompt_tokens = len(prompt.split()) * 1.3  # Rough estimate

            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                include_stop_str_in_output=False,
                skip_special_tokens=True,
                spaces_between_special_tokens=True,
            )

            # Generate with unique request ID
            request_id = f"req_{model_name}_{int(time.time() * 1000000)}"

            if stream:
                # Streaming generation
                outputs = []
                async for output in engine.generate(
                    prompt, sampling_params, request_id=request_id
                ):
                    outputs.append(output)

                final_output = outputs[-1] if outputs else None
            else:
                # Non-streaming generation
                outputs = await engine.generate(
                    prompt, sampling_params, request_id=request_id
                )
                final_output = outputs[0] if outputs else None

            if not final_output or not final_output.outputs:
                raise RuntimeError("No output generated")

            output = final_output.outputs[0]
            generated_text = output.text
            completion_tokens = len(output.token_ids)
            total_tokens = prompt_tokens + completion_tokens

            latency_ms = (time.time() - start_time) * 1000
            tokens_per_second = (
                completion_tokens / (latency_ms / 1000) if latency_ms > 0 else 0
            )

            result = InferenceResult(
                text=generated_text,
                success=True,
                latency_ms=latency_ms,
                tokens_generated=completion_tokens,
                model_used=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                tokens_per_second=tokens_per_second,
                finish_reason=output.finish_reason,
                cached=False,
            )

            # Cache result
            if use_cache and cache_key:
                self.cache[cache_key] = result
                # Limit cache size
                if len(self.cache) > 1000:
                    # Remove oldest entries
                    oldest_keys = list(self.cache.keys())[:100]
                    for key in oldest_keys:
                        del self.cache[key]

            # Update metrics
            if self.enable_metrics:
                self._update_metrics(model_name, latency_ms, completion_tokens, True)

            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = str(e)
            logger.error(f"Generation failed for {model_name}: {error_msg}")

            # Update error metrics
            if self.enable_metrics:
                self._update_metrics(model_name, latency_ms, 0, False)
                self.performance_metrics[model_name]["errors"] += 1

            return InferenceResult(
                text="",
                success=False,
                latency_ms=latency_ms,
                tokens_generated=0,
                model_used=model_name,
                error=error_msg,
            )

    def _get_cache_key(
        self, prompt: str, model_name: str, max_tokens: int, temperature: float
    ) -> str:
        """Generate cache key for request"""
        import hashlib

        content = f"{prompt}|{model_name}|{max_tokens}|{temperature}"
        return hashlib.md5(content.encode()).hexdigest()

    def _auto_select_model(self, prompt: str) -> str:
        """
        Enhanced model selection based on prompt analysis
        """
        prompt_lower = prompt.lower()

        # Advanced SQL detection
        sql_keywords = [
            "select",
            "from",
            "where",
            "join",
            "group by",
            "having",
            "order by",
            "insert",
            "update",
            "delete",
            "create table",
            "alter table",
            "sql",
            "query",
        ]
        if any(keyword in prompt_lower for keyword in sql_keywords):
            if "sqlcoder-7b" in self.models:
                return "sqlcoder-7b"

        # Advanced Python detection
        python_keywords = [
            "python",
            "def ",
            "class ",
            "import ",
            "from ",
            "__init__",
            "function",
            "lambda",
            "return",
            "if __name__",
            "pip install",
        ]
        if any(keyword in prompt_lower for keyword in python_keywords):
            # Prefer specialized code models
            for model in ["deepseek-coder-7b", "wizardcoder-7b", "codellama-7b"]:
                if model in self.models:
                    return model

        # Reasoning and analysis tasks
        reasoning_keywords = [
            "analyze",
            "reason",
            "explain",
            "why",
            "because",
            "compare",
            "evaluate",
            "assess",
            "conclude",
            "infer",
        ]
        if any(keyword in prompt_lower for keyword in reasoning_keywords):
            if "phi3-mini" in self.models:
                return "phi3-mini"

        # Default to general model based on performance
        preferred_order = [
            "llama3-8b",
            "phi3-mini",
            "codellama-7b",
            "deepseek-coder-7b",
            "wizardcoder-7b",
            "sqlcoder-7b",
        ]
        for model in preferred_order:
            if model in self.models:
                return model

        # Fallback to first available model
        return list(self.models.keys())[0] if self.models else "none"

    def _update_metrics(
        self, model_name: str, latency_ms: float, tokens: int, success: bool
    ):
        """Enhanced metrics tracking"""
        if not self.enable_metrics or model_name not in self.performance_metrics:
            return

        metrics = self.performance_metrics[model_name]
        metrics["total_requests"] += 1
        metrics["total_latency"] += latency_ms
        metrics["total_tokens"] += tokens
        metrics["avg_latency"] = metrics["total_latency"] / metrics["total_requests"]

        if latency_ms > 0 and tokens > 0:
            metrics["tokens_per_second"] = (tokens / latency_ms) * 1000

        # Calculate success rate
        if success:
            metrics["success_rate"] = (
                metrics["success_rate"] * (metrics["total_requests"] - 1) + 1
            ) / metrics["total_requests"]
        else:
            metrics["success_rate"] = (
                metrics["success_rate"] * (metrics["total_requests"] - 1)
            ) / metrics["total_requests"]

    def _update_gpu_stats(self):
        """Update GPU statistics"""
        if not torch.cuda.is_available():
            return

        try:
            device = torch.cuda.current_device()
            self.gpu_stats = {
                "memory_used": torch.cuda.memory_allocated(device) / 1024**3,
                "memory_total": torch.cuda.get_device_properties(device).total_memory
                / 1024**3,
                "memory_reserved": torch.cuda.memory_reserved(device) / 1024**3,
                "device_name": torch.cuda.get_device_properties(device).name,
                "device_count": torch.cuda.device_count(),
            }
            self.gpu_stats["utilization"] = (
                self.gpu_stats["memory_used"] / self.gpu_stats["memory_total"]
            ) * 100

        except Exception as e:
            logger.warning(f"Could not update GPU stats: {e}")

    async def batch_generate(
        self, prompts: list[str], model_name: str | None = None, **kwargs
    ) -> BatchInferenceResult:
        """
        High-performance batch generation with automatic batching
        """
        if not prompts:
            return BatchInferenceResult(
                results=[],
                total_latency_ms=0,
                throughput_tokens_per_second=0,
                success_rate=0,
            )

        if not self.is_initialized:
            error_result = InferenceResult(
                text="",
                success=False,
                latency_ms=0,
                tokens_generated=0,
                model_used="none",
                error="Model server not initialized",
            )
            return BatchInferenceResult(
                results=[error_result] * len(prompts),
                total_latency_ms=0,
                throughput_tokens_per_second=0,
                success_rate=0,
            )

        # Auto-select model for batch
        if model_name is None:
            model_name = self._auto_select_model(prompts[0])

        if model_name not in self.models:
            error_result = InferenceResult(
                text="",
                success=False,
                latency_ms=0,
                tokens_generated=0,
                model_used=model_name,
                error=f"Model {model_name} not available",
            )
            return BatchInferenceResult(
                results=[error_result] * len(prompts),
                total_latency_ms=0,
                throughput_tokens_per_second=0,
                success_rate=0,
            )

        config = self.model_configs[model_name]
        engine = self.models[model_name]

        start_time = time.time()

        try:
            # Create sampling params
            sampling_params = SamplingParams(
                temperature=kwargs.get("temperature", config.temperature),
                max_tokens=kwargs.get("max_tokens", 1024),
                top_p=kwargs.get("top_p", 0.9),
                frequency_penalty=kwargs.get("frequency_penalty", 0.1),
                presence_penalty=kwargs.get("presence_penalty", 0.1),
                stop=kwargs.get(
                    "stop", ["<|endoftext|>", "<|end|>", "</s>", "<|im_end|>"]
                ),
            )

            # Generate batch with unique request IDs
            batch_requests = []
            for i, prompt in enumerate(prompts):
                batch_requests.append(
                    {
                        "prompt": prompt,
                        "sampling_params": sampling_params,
                        "request_id": f"batch_{model_name}_{i}_{int(time.time() * 1000000)}",
                    }
                )

            # Execute batch generation
            batch_outputs = []
            for request in batch_requests:
                outputs = await engine.generate(
                    request["prompt"], request["sampling_params"], request["request_id"]
                )
                batch_outputs.append(outputs[0] if outputs else None)

            total_latency_ms = (time.time() - start_time) * 1000

            # Process results
            results = []
            total_tokens = 0
            successful_requests = 0

            for i, (prompt, output) in enumerate(zip(prompts, batch_outputs)):
                if output and output.outputs:
                    generated_text = output.outputs[0].text
                    completion_tokens = len(output.outputs[0].token_ids)

                    # Count prompt tokens
                    prompt_tokens = 0
                    if model_name in self.tokenizers:
                        try:
                            prompt_tokens = len(
                                self.tokenizers[model_name].encode(prompt)
                            )
                        except:
                            prompt_tokens = len(prompt.split()) * 1.3

                    total_tokens += completion_tokens
                    successful_requests += 1

                    result = InferenceResult(
                        text=generated_text,
                        success=True,
                        latency_ms=total_latency_ms,  # Shared latency for batch
                        tokens_generated=completion_tokens,
                        model_used=model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                        tokens_per_second=completion_tokens / (total_latency_ms / 1000)
                        if total_latency_ms > 0
                        else 0,
                        finish_reason=output.outputs[0].finish_reason,
                    )
                    results.append(result)

                    # Update metrics
                    if self.enable_metrics:
                        self._update_metrics(
                            model_name,
                            total_latency_ms / len(prompts),
                            completion_tokens,
                            True,
                        )
                else:
                    error_result = InferenceResult(
                        text="",
                        success=False,
                        latency_ms=total_latency_ms,
                        tokens_generated=0,
                        model_used=model_name,
                        error="No output generated",
                    )
                    results.append(error_result)

                    if self.enable_metrics:
                        self._update_metrics(
                            model_name, total_latency_ms / len(prompts), 0, False
                        )

            # Calculate batch metrics
            throughput = (
                (total_tokens / total_latency_ms) * 1000 if total_latency_ms > 0 else 0
            )
            success_rate = successful_requests / len(prompts) if prompts else 0

            return BatchInferenceResult(
                results=results,
                total_latency_ms=total_latency_ms,
                throughput_tokens_per_second=throughput,
                success_rate=success_rate,
            )

        except Exception as e:
            total_latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Batch generation failed for {model_name}: {e}")

            error_result = InferenceResult(
                text="",
                success=False,
                latency_ms=total_latency_ms,
                tokens_generated=0,
                model_used=model_name,
                error=str(e),
            )

            return BatchInferenceResult(
                results=[error_result] * len(prompts),
                total_latency_ms=total_latency_ms,
                throughput_tokens_per_second=0,
                success_rate=0,
            )

    async def _batch_processor(self):
        """Background batch processor for automatic batching"""
        while True:
            try:
                await asyncio.sleep(0.01)  # Small delay to allow batching
                # Implementation for automatic request batching
                # This would collect individual requests and batch them
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processor error: {e}")

    def get_model_info(self) -> dict[str, Any]:
        """
        Comprehensive model and system information
        """
        return {
            "initialized": self.is_initialized,
            "loaded_models": list(self.models.keys()),
            "model_configs": {
                name: {
                    "specialization": config.specialization,
                    "quantization": config.quantization,
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature,
                    "tensor_parallel_size": config.tensor_parallel_size,
                    "gpu_memory_utilization": config.gpu_memory_utilization,
                    "batch_size": config.max_batch_size,
                }
                for name, config in self.model_configs.items()
            },
            "performance_metrics": self.performance_metrics,
            "gpu_stats": self.gpu_stats,
            "cache_size": len(self.cache),
            "system_info": {
                "vllm_available": VLLM_AVAILABLE,
                "torch_available": TORCH_AVAILABLE,
                "cuda_available": torch.cuda.is_available()
                if TORCH_AVAILABLE
                else False,
                "gpu_count": torch.cuda.device_count()
                if TORCH_AVAILABLE and torch.cuda.is_available()
                else 0,
            },
        }

    async def unload_model(self, model_name: str):
        """
        Safely unload a model and free GPU memory
        """
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not loaded")
            return

        logger.info(f"Unloading model {model_name}...")

        try:
            # Remove from active models
            del self.models[model_name]

            if model_name in self.model_configs:
                del self.model_configs[model_name]

            if model_name in self.tokenizers:
                del self.tokenizers[model_name]

            if model_name in self.performance_metrics:
                del self.performance_metrics[model_name]

            # Clear cache entries for this model
            cache_keys_to_remove = [k for k in self.cache.keys() if model_name in k]
            for key in cache_keys_to_remove:
                del self.cache[key]

            # Force garbage collection and GPU memory cleanup
            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logger.info(f"Successfully unloaded model {model_name}")
            self._update_gpu_stats()

        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")

    async def reload_model(self, model_name: str):
        """
        Reload a specific model (useful for model updates)
        """
        if model_name in self.models:
            await self.unload_model(model_name)

        await self.initialize_models([model_name])

    async def shutdown(self):
        """
        Graceful shutdown with proper cleanup
        """
        logger.info("Shutting down model server...")

        # Cancel batch processor
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass

        # Unload all models
        model_names = list(self.models.keys())
        for model_name in model_names:
            await self.unload_model(model_name)

        # Clear cache
        self.cache.clear()

        self.is_initialized = False
        logger.info("Model server shutdown complete")

    async def health_check(self) -> dict[str, Any]:
        """
        Comprehensive health check
        """
        health_status = {
            "status": "healthy" if self.is_initialized else "unhealthy",
            "models_loaded": len(self.models),
            "models_status": {},
            "gpu_status": self.gpu_stats,
            "cache_size": len(self.cache),
            "timestamp": time.time(),
        }

        # Test each model with a simple prompt
        for model_name in self.models:
            try:
                test_result = await self.generate(
                    "Test prompt", model_name=model_name, max_tokens=10, use_cache=False
                )
                health_status["models_status"][model_name] = {
                    "status": "healthy" if test_result.success else "unhealthy",
                    "error": test_result.error,
                    "latency_ms": test_result.latency_ms,
                }
            except Exception as e:
                health_status["models_status"][model_name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "latency_ms": 0,
                }

        return health_status


# Global model server instance
model_server = AdvancedModelServer(enable_metrics=True)

# Alias for backward compatibility
ModelServer = AdvancedModelServer
