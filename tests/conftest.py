"""Shared fixtures: a miniature but structurally faithful Qwen3 model directory.

The fast tests never download weights. Instead they build a ~40 KiB checkpoint on
disk with the exact layout ``tools/convert_model.py`` produces (``manifest.json``
plus one ``.npy`` per tensor) and a real byte-level BPE tokenizer trained on a
few Chinese/emoji/English sentences, so the loader, the mmap path, the chat
template and the incremental decoder are all exercised for real.

The tiny config keeps Qwen3's awkward dimension relationships rather than
simplifying them:

* ``head_dim * num_attention_heads != hidden_size`` (8 * 4 = 32 != 16), so any
  code that infers ``head_dim`` from ``hidden_size`` fails here.
* ``num_attention_heads / num_key_value_heads == 2``, the same GQA group size as
  Qwen3-0.6B.
* ``tie_word_embeddings = true``, so ``lm_head.weight`` exists only as an alias.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from tokenizers import AddedToken
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

#: Deliberately *not* the official Qwen3 template, but it exercises the same
#: features: string concatenation, the ``enable_thinking`` branch, and
#: ``add_generation_prompt``. Rendering fidelity against the real template is
#: checked by tools/validate_reference.py.
TINY_CHAT_TEMPLATE = """\
{%- for message in messages %}
{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] }}
{%- if message['role'] == 'assistant' and enable_thinking is defined and enable_thinking is false %}
{{- '\\n<think>\\n\\n</tool_response>' }}
{%- endif %}
{{- '<|im_end|>\\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\\n' }}
{%- endif %}"""

TINY_CORPUS = [
    "你好，世界！SlowQ 用 CPU 慢慢炖出 token。",
    "hello world, the quick brown fox jumps over the lazy dog",
    "🐢🍲 慢炖 emoji test 😀🚀 中文测试",
] * 4

#: Every marker the template emits must be registered with ``special=True``;
#: that is what lets ``decode(skip_special_tokens=True)`` strip it, exactly as
#: the official tokenizer.json strips a trailing stop token. The chat markers are
#: scraped from the template itself so the two can never drift apart.
SPECIAL_TOKENS = [
    *sorted(set(re.findall(r"<\|[A-Za-z_]+\|>", TINY_CHAT_TEMPLATE))),
    "<|fim_middle|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "@Repository",
]

TINY_ARCHITECTURE = {
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "max_position_embeddings": 256,
    "hidden_act": "silu",
    "tie_word_embeddings": True,
    "attention_bias": False,
    "use_sliding_window": False,
}


# --------------------------------------------------------------------------- #
# tokenizer
# --------------------------------------------------------------------------- #


def build_tiny_tokenizer(root: Path) -> dict[str, int]:
    """Train a byte-level BPE tokenizer and write it plus its config files."""
    tokenizer = HFTokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    # ``special=True`` matters: it is what makes ``decode(skip_special_tokens=True)``
    # strip the chat markers from generated text, as the real Qwen3 tokenizer does.
    trainer = BpeTrainer(
        vocab_size=512,
        min_frequency=1,
        show_progress=False,
        initial_alphabet=ByteLevel.alphabet(),
        special_tokens=[AddedToken(token, special=True, normalized=False) for token in SPECIAL_TOKENS],
    )
    tokenizer.train_from_iterator(TINY_CORPUS, trainer)
    tokenizer.save(str(root / "tokenizer.json"))

    ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": TINY_CHAT_TEMPLATE,
                "bos_token": None,
                # Qwen3 stops on <|fim_middle|>; <|fim_prefix|> is the other published eos id.
                "eos_token": "<|fim_middle|>",
                "pad_token": "<|fim_middle|>",
                "unk_token": None,
                "add_bos_token": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "generation_config.json").write_text(
        json.dumps(
            {
                # Mirrors Qwen3-0.6B, which publishes two stop ids.
                "eos_token_id": [ids["<|fim_middle|>"], ids["<|fim_prefix|>"]],
                "pad_token_id": ids["<|fim_middle|>"],
                "bos_token_id": None,
                "temperature": 0.7,
                "top_k": 20,
                "top_p": 0.8,
                "do_sample": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return ids


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #


def tiny_tensors(
    vocab_size: int,
    hidden_size: int = 16,
    intermediate_size: int = 32,
    num_layers: int = 2,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 8,
    seed: int = 20261018,
) -> dict[str, np.ndarray]:
    """Random FP32 tensors keyed by their canonical HF names.

    Projections use ``std=0.02`` like a freshly initialised transformer; norm
    weights use ``N(1, 0.02)`` rather than all-ones so a bug that ignores the
    norm weight cannot hide.
    """
    rng = np.random.default_rng(seed)
    q_out = num_heads * head_dim
    kv_out = num_kv_heads * head_dim

    def matrix(out_features: int, in_features: int) -> np.ndarray:
        return rng.normal(0.0, 0.02, size=(out_features, in_features)).astype(np.float32)

    def norm_weight(size: int) -> np.ndarray:
        return rng.normal(1.0, 0.02, size=(size,)).astype(np.float32)

    tensors = {
        "model.embed_tokens.weight": matrix(vocab_size, hidden_size),
        "model.norm.weight": norm_weight(hidden_size),
    }
    for i in range(num_layers):
        p = f"model.layers.{i}"
        tensors[f"{p}.self_attn.q_proj.weight"] = matrix(q_out, hidden_size)
        tensors[f"{p}.self_attn.k_proj.weight"] = matrix(kv_out, hidden_size)
        tensors[f"{p}.self_attn.v_proj.weight"] = matrix(kv_out, hidden_size)
        tensors[f"{p}.self_attn.o_proj.weight"] = matrix(hidden_size, q_out)
        tensors[f"{p}.self_attn.q_norm.weight"] = norm_weight(head_dim)
        tensors[f"{p}.self_attn.k_norm.weight"] = norm_weight(head_dim)
        tensors[f"{p}.mlp.gate_proj.weight"] = matrix(intermediate_size, hidden_size)
        tensors[f"{p}.mlp.up_proj.weight"] = matrix(intermediate_size, hidden_size)
        tensors[f"{p}.mlp.down_proj.weight"] = matrix(hidden_size, intermediate_size)
        tensors[f"{p}.input_layernorm.weight"] = norm_weight(hidden_size)
        tensors[f"{p}.post_attention_layernorm.weight"] = norm_weight(hidden_size)
    return tensors


def write_tiny_model(
    root: Path,
    *,
    vocab_size: int,
    omit: tuple[str, ...] = (),
    overrides: dict[str, np.ndarray] | None = None,
    config_overrides: dict | None = None,
) -> Path:
    """Write a complete converted-model directory and return its path."""
    root = Path(root)
    (root / "tensors").mkdir(parents=True, exist_ok=True)

    config = dict(TINY_ARCHITECTURE, vocab_size=vocab_size)
    if config_overrides:
        config.update(config_overrides)
    (root / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    tensors = tiny_tensors(vocab_size=vocab_size)
    if overrides:
        tensors.update(overrides)

    entries = []
    aliases: dict[str, str] = {}
    for name in sorted(tensors):
        if name in omit:
            continue
        array = np.ascontiguousarray(tensors[name], dtype=np.float32)
        relative = f"{name}.npy"
        np.save(root / "tensors" / relative, array)
        entries.append(
            {
                "name": name,
                "file": relative,
                "shape": list(array.shape),
                "dtype": "float32",
                "source_dtype": "BF16",
            }
        )

    if config["tie_word_embeddings"]:
        # The converter stores a tied lm_head once and records the alias.
        aliases["lm_head.weight"] = "model.embed_tokens.weight"

    manifest = {
        "format_version": 1,
        "created_at": "2026-10-18T00:00:00+0000",
        "tool": "tests/conftest.py",
        "compute_dtype": "float32",
        "source": {
            "repo_id": "tiny:local",
            "revision": "0" * 40,
            "local_path": str(root),
            "files": [{"name": "model.safetensors", "sha256": "0" * 64}],
        },
        "weight_layout": "[out_features, in_features]; runtime computes x @ W.T",
        "tensors": entries,
        "aliases": aliases,
        "tokenizer_files": ["tokenizer.json", "tokenizer_config.json"],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return root


# --------------------------------------------------------------------------- #
# sampler stand-in
# --------------------------------------------------------------------------- #


def make_scripted_sampler(token_ids: Sequence[int]):
    """A ``Sampler`` stand-in that replays a fixed token sequence.

    The tiny model has random weights, so which token it would emit is not
    something a test can arrange. Scripting the sampler pins down the *loop*
    instead: stop conditions, streaming, text assembly, and CLI output routing.
    """

    class Scripted:
        def __init__(self, params=None) -> None:
            self.params = params
            self.calls = 0

        @property
        def is_greedy(self) -> bool:
            return True

        def sample(self, logits) -> int:
            token_id = token_ids[min(self.calls, len(token_ids) - 1)]
            self.calls += 1
            return int(token_id)

    return Scripted


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A complete tiny model directory, built once per test session."""
    root = tmp_path_factory.mktemp("tiny-model") / "tiny-qwen3"
    root.mkdir()
    build_tiny_tokenizer(root)
    vocab_size = HFTokenizer.from_file(str(root / "tokenizer.json")).get_vocab_size()
    write_tiny_model(root, vocab_size=vocab_size)
    return root


@pytest.fixture
def write_model():
    """Factory for standalone weight directories (no tokenizer, no fixtures)."""
    return write_tiny_model


@pytest.fixture
def random_tensors():
    """Factory for the random tensors ``write_tiny_model`` would have written."""
    return tiny_tensors


@pytest.fixture
def tiny_config(tiny_model_dir: Path):
    from slowq.config import ModelConfig

    return ModelConfig.from_json_file(tiny_model_dir / "config.json")


@pytest.fixture
def tiny_engine(tiny_model_dir: Path):
    """A fresh Engine per test: the KV cache is shared state, so never reuse one."""
    from slowq import Engine

    return Engine.from_directory(tiny_model_dir, max_context=64)


@pytest.fixture
def tiny_tokenizer(tiny_model_dir: Path):
    from slowq.tokenizer import Tokenizer

    return Tokenizer.from_directory(tiny_model_dir)


@pytest.fixture
def scripted_sampler():
    """Factory for a deterministic ``Sampler`` replacement (see conftest)."""
    return make_scripted_sampler
