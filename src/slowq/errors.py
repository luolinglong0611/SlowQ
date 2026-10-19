"""Exception hierarchy for SlowQ.

One base class plus one per way SlowQ refuses on purpose: unsupported model
configuration, bad model data, an input feature v0.1 lacks, an illegal sampling
request, and a request that does not fit the context window. Anything else is a
bug and should stay a plain Python exception.
"""

from __future__ import annotations


class SlowQError(Exception):
    """Base class for all errors raised deliberately by SlowQ."""


class ConfigError(SlowQError):
    """The model configuration is unsupported, malformed, or self-inconsistent."""


class WeightError(SlowQError):
    """A weight file is missing, unreadable, or has an unexpected shape/dtype."""


class UnsupportedInputError(SlowQError):
    """The request uses a feature v0.1 does not implement.

    Raised for tool calls, tool results, and structured/multimodal content so a
    caller gets an explanation instead of silently dropped fields.
    """


class SamplingError(SlowQError):
    """Sampling parameters are invalid, or the logits cannot be sampled from.

    Covers illegal temperature/top-k/top-p, non-finite logits, and a top-k/top-p
    filter that emptied the candidate set. Sampling something arbitrary here
    would look like a model bug rather than a bad request.
    """


class ContextOverflowError(SlowQError):
    """The request does not fit inside the configured context window.

    SlowQ never silently truncates a prompt and never silently overwrites live
    KV cache entries, so this is always raised instead.
    """
