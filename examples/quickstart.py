#!/usr/bin/env python3
"""The whole public API on one screen.

    python examples/quickstart.py --model models/Qwen3-0.6B-slowq

Four things, in the order you would actually need them: load a converted
checkpoint, generate one answer, stream an answer token by token, and read the
timings back out. Nothing here is required — ``Engine.generate`` alone is enough
to use SlowQ — but this is the shortest path from a directory of ``.npy`` files
to text on the terminal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slowq import Engine, SamplingParams


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="models/Qwen3-0.6B-slowq")
    parser.add_argument("--prompt", default="用一句话解释什么是 KV cache。")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    # 1. Load. Fully offline: this reads config.json, manifest.json and mmaps the
    #    weights. No network access, no torch, no transformers.
    engine = Engine.from_directory(args.model, max_context=1024)
    print(f"loaded in {engine.load_seconds:.2f} s", file=sys.stderr)

    messages = [{"role": "user", "content": args.prompt}]

    # 2. One blocking call. temperature=0 means greedy, which is reproducible.
    result = engine.generate(messages, max_new_tokens=args.max_new_tokens)
    print("\n--- generate ---")
    print(result.text)
    print(f"\nstopped because: {result.stop_reason.value}")
    print(f"prompt {result.metrics.prompt_tokens} tokens -> "
          f"{result.metrics.generated_tokens} generated")

    # 3. Streaming. Each step carries only the new text, so a partially decoded
    #    multi-byte character is held back rather than emitted as U+FFFD.
    print("\n--- stream ---")
    for step in engine.stream(messages, max_new_tokens=args.max_new_tokens):
        print(step.text_delta, end="", flush=True)
    print()

    # 4. Sampling instead of greedy. A fixed seed replays the same tokens.
    sampled = engine.generate(
        messages,
        sampling=SamplingParams(temperature=0.7, top_k=20, top_p=0.8, seed=20261018),
        max_new_tokens=args.max_new_tokens,
    )
    print("\n--- sampled (temperature 0.7, seed 20261018) ---")
    print(sampled.text)

    # 5. The numbers. Every interval is defined in docs/architecture.md.
    print("\n--- metrics ---")
    for label, value in result.metrics.report():
        print(f"  {label:<24} {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
