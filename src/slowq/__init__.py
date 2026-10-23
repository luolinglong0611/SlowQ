"""慢炖 · SlowQ — small-model inference on CPU with NumPy only.

Public API::

    from slowq import Engine, SamplingParams

    engine = Engine.from_directory("models/Qwen3-0.6B-slowq")
    result = engine.generate(prompt="你好，介绍一下你自己")
    print(result.text, result.metrics.decode_tokens_per_second)

Three tasks are supported, and which one a converted directory holds is recorded
in its manifest rather than guessed from the checkpoint's architecture name::

    from slowq import Engine, EmbeddingEngine

    Engine.from_directory(...)           # text generation (qwen3 / qwen2 / llama)
    EmbeddingEngine.from_directory(...)  # text embedding (Qwen3-Embedding)

The third, image+text generation, lives in :mod:`slowq.vlm` and is imported
separately because it is the only part of SlowQ that can need Pillow::

    from slowq.vlm import VisionEngine

    VisionEngine.from_directory(...)     # image+text generation (SmolVLM)

The runtime import graph is numpy + tokenizers + Jinja2 only, plus Pillow for
``slowq.vlm`` and only when an image is actually read. Nothing here imports
torch, transformers or safetensors; those live in the optional ``slowq[ref]``
extra and are used solely by ``tools/validate_reference.py``.
"""

from __future__ import annotations

__version__ = "0.2.0"

from .cache import KVCache
from .config import (
    FAMILY_IDEFICS3,
    FAMILY_LLAMA,
    FAMILY_QWEN2,
    FAMILY_QWEN3,
    TASK_IMAGE_TEXT_GENERATION,
    TASK_TEXT_EMBEDDING,
    TASK_TEXT_GENERATION,
    GenerationConfig,
    ModelConfig,
    TextFamily,
)
from .embedding import EmbeddingEngine, EmbeddingModel, EmbeddingResult
from .engine import (
    DEFAULT_MAX_CONTEXT,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_PREFILL_CHUNK_SIZE,
    Engine,
    GenerationResult,
    GenerationStep,
    StopReason,
    recorded_task,
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
from .model import CausalLMModel, DebugProbe, DecoderModel, DecoderOutput, ForwardResult
from .sampling import Sampler, SamplingParams
from .tokenizer import ChatMessage, IncrementalTextDecoder, Tokenizer
from .weights import Manifest, TensorSpec, Weights

__all__ = [
    "__version__",
    "CausalLMModel",
    "ChatMessage",
    "ConfigError",
    "ContextOverflowError",
    "DEFAULT_MAX_CONTEXT",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_PREFILL_CHUNK_SIZE",
    "DebugProbe",
    "DecoderModel",
    "DecoderOutput",
    "EmbeddingEngine",
    "EmbeddingModel",
    "EmbeddingResult",
    "Engine",
    "FAMILY_IDEFICS3",
    "FAMILY_LLAMA",
    "FAMILY_QWEN2",
    "FAMILY_QWEN3",
    "ForwardResult",
    "GenerationConfig",
    "GenerationMetrics",
    "GenerationResult",
    "GenerationStep",
    "IncrementalTextDecoder",
    "KVCache",
    "Manifest",
    "ModelConfig",
    "Sampler",
    "SamplingError",
    "SamplingParams",
    "SlowQError",
    "StopReason",
    "TASK_IMAGE_TEXT_GENERATION",
    "TASK_TEXT_EMBEDDING",
    "TASK_TEXT_GENERATION",
    "TensorSpec",
    "TextFamily",
    "Tokenizer",
    "UnsupportedInputError",
    "WeightError",
    "Weights",
    "environment_info",
    "peak_rss_bytes",
    "recorded_task",
]
