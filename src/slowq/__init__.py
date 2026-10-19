"""慢炖 · SlowQ — Qwen3-0.6B inference on CPU with NumPy only.

Public API::

    from slowq import Engine, SamplingParams

    engine = Engine.from_directory("models/Qwen3-0.6B-slowq")
    result = engine.generate(prompt="你好，介绍一下你自己")
    print(result.text, result.metrics.decode_tokens_per_second)

The runtime import graph is numpy + tokenizers + Jinja2 only. Nothing here
imports torch, transformers or safetensors; those live in the optional
``slowq[ref]`` extra and are used solely by ``tools/validate_reference.py``.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .cache import KVCache
from .config import GenerationConfig, ModelConfig
from .engine import (
    DEFAULT_MAX_CONTEXT,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_PREFILL_CHUNK_SIZE,
    Engine,
    GenerationResult,
    GenerationStep,
    StopReason,
)
from .errors import (
    ConfigError,
    ContextOverflowError,
    SamplingError,
    SlowQError,
    UnsupportedInputError,
    WeightError,
)
from .metrics import GenerationMetrics, environment_info, peak_rss_bytes
from .model import DebugProbe, ForwardResult, Qwen3Model
from .sampling import Sampler, SamplingParams
from .tokenizer import ChatMessage, IncrementalTextDecoder, Tokenizer
from .weights import Manifest, TensorSpec, Weights

__all__ = [
    "__version__",
    "ChatMessage",
    "ConfigError",
    "ContextOverflowError",
    "DEFAULT_MAX_CONTEXT",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_PREFILL_CHUNK_SIZE",
    "DebugProbe",
    "Engine",
    "ForwardResult",
    "GenerationConfig",
    "GenerationMetrics",
    "GenerationResult",
    "GenerationStep",
    "IncrementalTextDecoder",
    "KVCache",
    "Manifest",
    "ModelConfig",
    "Qwen3Model",
    "Sampler",
    "SamplingError",
    "SamplingParams",
    "SlowQError",
    "StopReason",
    "TensorSpec",
    "Tokenizer",
    "UnsupportedInputError",
    "WeightError",
    "Weights",
    "environment_info",
    "peak_rss_bytes",
]
