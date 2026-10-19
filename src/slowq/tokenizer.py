"""Tokenizer, official chat template rendering, and incremental text decoding.

Two deliberate choices:

* The vocabulary comes from ``tokenizers`` + the official ``tokenizer.json``; we
  never reimplement BPE. Qwen's post-processor is ``ByteLevel`` with
  ``add_prefix_space=False``, so ``add_special_tokens`` adds nothing — we pass
  ``False`` anyway to make "the chat template emits every special token itself"
  explicit and to rule out double-adding ``<|fim_prefix|>``/``<|fim_middle|>``.
* The prompt is rendered with Jinja2 from the ``chat_template`` string in
  ``tokenizer_config.json``, using the same environment settings as
  ``transformers`` (``trim_blocks`` + ``lstrip_blocks`` matter: the official
  template is indented). We do not hand-write a template and do not guess
  BOS/EOS.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer as _HFTokenizer
from tokenizers import __version__ as _TOKENIZERS_VERSION

from .config import GenerationConfig
from .errors import ConfigError, UnsupportedInputError

TOKENIZER_FILE = "tokenizer.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

#: v0.1 renders text-only conversations. ``enable_thinking=False`` selects the
#: non-thinking branch of the official Qwen3 template.
SUPPORTED_ROLES = ("system", "user", "assistant")
DEFAULT_ENABLE_THINKING = False


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def to_template_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


def normalize_messages(messages: Iterable[Mapping[str, Any] | ChatMessage]) -> list[ChatMessage]:
    """Validate and convert user-supplied messages.

    Rejects anything v0.1 cannot represent: tool calls, tool results, and
    structured/multimodal content (a list of parts instead of a string).
    """
    normalized: list[ChatMessage] = []
    materialized = list(messages)
    if not materialized:
        raise UnsupportedInputError("messages must not be empty")

    for index, message in enumerate(materialized):
        where = f"messages[{index}]"
        if isinstance(message, ChatMessage):
            raw: Mapping[str, Any] = asdict(message)
        elif isinstance(message, Mapping):
            raw = message
        else:
            raise UnsupportedInputError(
                f"{where} must be a mapping or ChatMessage, got {type(message).__name__}"
            )

        role = raw.get("role")
        if role not in SUPPORTED_ROLES:
            raise UnsupportedInputError(
                f"{where}.role={role!r} is not supported in v0.1; "
                f"expected one of {list(SUPPORTED_ROLES)}. Tool calls and tool "
                "results are out of scope."
            )
        for rejected in ("tool_calls", "function_call"):
            if raw.get(rejected):
                raise UnsupportedInputError(
                    f"{where}.{rejected} is not supported in SlowQ v0.1."
                )
        content = raw.get("content")
        if not isinstance(content, str):
            raise UnsupportedInputError(
                f"{where}.content must be a plain string, got {type(content).__name__}. "
                "SlowQ v0.1 is text-only: images, audio and multi-part content "
                "are not supported."
            )
        unknown = set(raw) - {"role", "content"}
        if unknown:
            raise UnsupportedInputError(
                f"{where} has unsupported field(s) {sorted(unknown)}; v0.1 accepts "
                "only 'role' and 'content'."
            )
        normalized.append(ChatMessage(role=str(role), content=content))
    return normalized


class IncrementalTextDecoder:
    """Streaming decode that never splits a multi-byte character.

    Chinese characters and emoji routinely span several byte-level BPE tokens, so
    decoding each token on its own and concatenating the strings yields ``锟斤拷``
    style garbage. Instead we buffer tokens and only emit when the decoded text
    no longer ends in the UTF-8 replacement character, which is how
    ``tokenizers``' ByteLevel decoder signals an incomplete trailing sequence.
    """

    #: Safety valve: a genuinely invalid (not merely incomplete) byte sequence
    #: would otherwise pin the buffer forever.
    MAX_HELD_TOKENS = 8

    def __init__(self, tokenizer: "Tokenizer", skip_special_tokens: bool = True) -> None:
        self._tokenizer = tokenizer
        self._skip_special_tokens = skip_special_tokens
        self._buffer: list[int] = []

    def push(self, token_id: int) -> str:
        self._buffer.append(int(token_id))
        text = self._tokenizer.decode(self._buffer, self._skip_special_tokens)
        if text.endswith("\ufffd") and len(self._buffer) < self.MAX_HELD_TOKENS:
            return ""
        self._buffer.clear()
        return text

    def push_many(self, token_ids: Iterable[int]) -> str:
        emitted: list[str] = []
        for token_id in token_ids:
            emitted.append(self.push(token_id))
        return "".join(emitted)

    def flush(self) -> str:
        """Emit anything still buffered (call at end of generation)."""
        text = self._tokenizer.decode(self._buffer, self._skip_special_tokens)
        self._buffer.clear()
        return text


class Tokenizer:
    """Thin wrapper over ``tokenizers`` plus the official Qwen3 chat template."""

    def __init__(
        self,
        tokenizer: _HFTokenizer,
        chat_template: str,
        special_token_ids: Mapping[str, int | None],
    ) -> None:
        self._tokenizer = tokenizer
        self._chat_template = chat_template
        self._template = self._compile_template(chat_template)
        self.special_token_ids = dict(special_token_ids)

    # ---- construction -------------------------------------------------------

    @classmethod
    def from_directory(cls, model_dir: str | Path) -> "Tokenizer":
        root = Path(model_dir)
        tokenizer_path = root / TOKENIZER_FILE
        config_path = root / TOKENIZER_CONFIG_FILE
        if not tokenizer_path.is_file():
            raise ConfigError(f"{tokenizer_path} not found in the model directory")
        if not config_path.is_file():
            raise ConfigError(f"{config_path} not found in the model directory")

        tokenizer_config = json.loads(config_path.read_text(encoding="utf-8"))
        chat_template = tokenizer_config.get("chat_template")
        if not isinstance(chat_template, str) or not chat_template.strip():
            raise ConfigError(
                f"{config_path} has no usable 'chat_template'. SlowQ will not "
                "guess a prompt format."
            )

        tokenizer = _HFTokenizer.from_file(str(tokenizer_path))
        tokenizer.no_truncation()
        tokenizer.no_padding()

        def _token_id(key: str) -> int | None:
            token = tokenizer_config.get(key)
            if token is None:
                return None
            token_id = tokenizer.token_to_id(token)
            if token_id is None:
                raise ConfigError(
                    f"tokenizer_config.json declares {key}={token!r}, which is not "
                    "in the vocabulary"
                )
            return int(token_id)

        return cls(
            tokenizer=tokenizer,
            chat_template=chat_template,
            special_token_ids={
                "bos": _token_id("bos_token"),
                "eos": _token_id("eos_token"),
                "pad": _token_id("pad_token"),
                "unk": _token_id("unk_token"),
            },
        )

    @staticmethod
    def _compile_template(chat_template: str) -> Any:
        def raise_exception(message: str) -> None:
            raise UnsupportedInputError(f"chat template error: {message}")

        def tojson(value: Any, indent: int | None = None) -> str:
            if isinstance(value, str):
                return value
            return json.dumps(value, ensure_ascii=False, indent=indent)

        # Same settings transformers uses to compile chat templates: the
        # official Qwen3 template is indented, so trim/lstrip_blocks decide
        # whether stray newlines end up in the prompt.
        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
        )
        environment.filters["tojson"] = tojson
        environment.globals["raise_exception"] = raise_exception
        environment.globals["tojson"] = tojson
        return environment.from_string(chat_template)

    # ---- encoding -----------------------------------------------------------

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Token IDs for ``text``. Special tokens already present are kept as-is."""
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens)

    def incremental_decoder(self, skip_special_tokens: bool = True) -> IncrementalTextDecoder:
        return IncrementalTextDecoder(self, skip_special_tokens=skip_special_tokens)

    # ---- chat template ------------------------------------------------------

    def render_chat(
        self,
        messages: Iterable[Mapping[str, Any] | ChatMessage],
        add_generation_prompt: bool = True,
        enable_thinking: bool = DEFAULT_ENABLE_THINKING,
    ) -> str:
        """Render the official Qwen3 chat template.

        ``enable_thinking=False`` (the v0.1 default) makes the template emit the
        empty ``<think>\n\n</think>`` marker, i.e. plain non-thinking mode.
        """
        normalized = normalize_messages(messages)
        rendered = self._template.render(
            messages=[m.to_template_dict() for m in normalized],
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            tools=[],
        )
        if not isinstance(rendered, str):
            raise ConfigError("chat template did not render to a string")
        return rendered

    def encode_chat(
        self,
        messages: Iterable[Mapping[str, Any] | ChatMessage],
        add_generation_prompt: bool = True,
        enable_thinking: bool = DEFAULT_ENABLE_THINKING,
    ) -> list[int]:
        return self.encode(
            self.render_chat(
                messages,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
            )
        )

    # ---- introspection ------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> int | None:
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> str | None:
        return self._tokenizer.id_to_token(token_id)

    def describe(self) -> list[tuple[str, Any]]:
        return [
            ("backend", f"tokenizers {_TOKENIZERS_VERSION}"),
            ("vocab size (incl. added tokens)", self.vocab_size),
            ("chat template", f"{len(self._chat_template)} chars from tokenizer_config.json"),
            ("default enable_thinking", DEFAULT_ENABLE_THINKING),
            *self.special_token_ids.items(),
        ]


def stop_token_ids(generation_config: GenerationConfig, tokenizer: Tokenizer) -> frozenset[int]:
    """Stop set taken from the official ``generation_config.json``, range-checked.

    Qwen3-0.6B publishes ``eos_token_id: [151645, 151643]`` (``<|fim_suffix|>`` and
    ``<|fim_prefix|>``); we use exactly that set rather than assuming a single EOS.
    """
    ids = frozenset(generation_config.eos_token_ids)
    if not ids:
        raise ConfigError("generation config declares no eos_token_id")
    out_of_range = sorted(t for t in ids if not 0 <= t < tokenizer.vocab_size)
    if out_of_range:
        raise ConfigError(
            f"eos_token_id(s) {out_of_range} are outside the tokenizer vocabulary "
            f"[0, {tokenizer.vocab_size})"
        )
    return ids
