#!/usr/bin/env python3
"""Price the sigmoid rewrite against the layer it lives in.

    .venv/bin/python tools/perf/sigmoid_check.py --out <dir>/sigmoid-check.json

``ops.sigmoid``'s docstring used to claim the masked form it replaced made sigmoid
"cost about as much as every matmul in the layer combined" on a 128-token prefill
chunk of Qwen3-0.6B. This script is the artefact behind that sentence, and the
first version of it did not support the sentence. Two things were wrong, and both
are fixed here rather than papered over:

* The input it called "real gate activation" was
  ``rms_norm(embed_tokens(ids), input_layernorm.weight) @ gate_proj.weight.T``.
  A decoder layer does not compute that. Between the embedding and ``gate_proj``
  sit the entire attention block, a residual add, and a *second* norm —
  ``post_attention_layernorm``, a different tensor from ``input_layernorm``. So
  the headline numbers (7.20x, 38.8% negative, 153% of the layer) priced an
  activation this model never produces. :func:`capture_layer_zero` now runs a real
  forward and records what ``silu`` is actually handed.
* The 153% was the operator's time divided by the sum of seven separately timed
  warm matmuls. That denominator is not a layer: it has no attention softmax, no
  norms, no residual adds, no elementwise work, and it was never measured inside a
  layer execution. :func:`substitution` replaces it with the only comparison that
  is an attribution — run the real layer twice, changing ``silu`` and nothing else.

Four quantities are reported and they are not interchangeable; see ``QUANTITIES``.

Nothing here is imported by the shipped package. It reads the measured tree's
``ops`` and ``model`` modules and carries the old implementation verbatim, so the
"before" side cannot drift when the "after" side is edited.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
# tests/test_perf_tooling.py loads every tool in tools/perf by path, which gives a
# module no package context, so a sibling import needs this line to resolve at all.
sys.path.insert(0, str(TOOL_DIR))

import out_guard  # noqa: E402  (path set above)
from slowq import ops  # noqa: E402
from slowq.model import CausalLMModel  # noqa: E402

DEFAULT_MODEL = ROOT / "models" / "Qwen3-0.6B-slowq"

#: The four numbers this script can produce, named so a reader cannot mistake one
#: for another. The published round-1 report presented (1)/(2) as if it were (3):
#: "old sigmoid is 153% of the layer" divided an isolated operator by a sum of
#: warm matmul medians and called the quotient a share of a real layer.
QUANTITIES = {
    "operator": "sigmoid 算子单独计时（新旧写法，同一份输入）",
    "matmul_sum": "该层 7 个 matmul 各自中位数之和（不是整层 profile）",
    "real_substitution": "真实前向里只替换 silu 的新旧写法（整层归因）",
    "end_to_end": "端到端收益，来自 ab/ 的 workloads，本脚本不测",
}

#: Synthetic inputs, kept on purpose: they bound how much the masked form's cost
#: depends on where the values sit relative to zero, which is the mechanism behind
#: the whole optimisation. They are *not* measurements of this model and must not
#: be quoted as attributions. The captured activation is deliberately absent from
#: this dict — it is the real thing, not a distribution.
SYNTHETIC_DISTRIBUTIONS = {
    "standard normal": "对称跨 0，给 masked 写法一个最坏情况上界",
    "all positive": "全部走 x>=0 分支：masked 写法在这里几乎不花钱",
    "all negative": "全部走 x<0 分支，同上",
}


def old_sigmoid(x: np.ndarray) -> np.ndarray:
    """The masked form that shipped before this round, verbatim."""
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = np.float32(1.0) / (np.float32(1.0) + np.exp(-x[positive], dtype=np.float32))
    exp_negative = np.exp(x[~positive], dtype=np.float32)
    out[~positive] = exp_negative / (np.float32(1.0) + exp_negative)
    return out


def old_silu(x: np.ndarray) -> np.ndarray:
    """``silu`` as it behaved before the rewrite, for the substitution run."""
    x = ops.as_f32(x)
    return x * old_sigmoid(x)


def timeit(fn: Callable[[], Any], repeat: int, warmup: int) -> dict[str, Any]:
    """Median/min/max plus every sample, so a reader can see the spread.

    ``repeat`` here means ``repeat`` independent calls of ``fn`` in one process.
    For the operator and the layer that is legitimate: each call starts from the
    same input and writes to a fresh output. It is *not* a cross-process figure,
    and the report says so.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeat):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    return {
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "mean": statistics.fmean(samples),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else None,
        "samples": samples,
        "repeat": repeat,
        "warmup": warmup,
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_sha256(path: Path) -> str:
    """Hash of what the file *does*, ignoring every docstring.

    Both hashes are recorded, and the difference is the point. The raw file hash
    answers "is this the exact file that ran?" — the strictest question, and the one
    that goes stale the moment anyone corrects a sentence in a docstring. This one
    answers "would it compute the same thing?", which is what a timing actually
    depends on.

    Without it the correction loop cannot close: the numbers measured here are
    quoted in ``ops.sigmoid``'s docstring, which lives *inside* one of the hashed
    files, so updating the prose to match the measurement invalidates the hash the
    measurement recorded, which then demands a new measurement, which moves the
    numbers again. Recording the docstring-insensitive hash breaks that cycle
    without weakening anything — an edit to ``sigmoid``'s body still changes it.

    ``ast.dump`` is used with attributes off (the default), so line numbers do not
    enter the hash either: rewrapping a docstring shifts every line below it.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                first.value.value = ""
    return hashlib.sha256(ast.dump(tree).encode("utf-8")).hexdigest()


def load_tensors(model_dir: Path, name: str) -> np.ndarray:
    return np.load(model_dir / "tensors" / f"{name}.npy", mmap_mode="r")


# --------------------------------------------------------------------------- #
# capture: what silu really receives
# --------------------------------------------------------------------------- #


def capture_layer_zero(model_dir: Path | str, input_ids: Sequence[int], layer: int = 0) -> dict[str, Any]:
    """Run one real forward and record the tensors the MLP is actually called with.

    ``silu_input`` is the argument to ``silu`` — the ``gate_proj`` output, which is
    what the operator microbenchmark must be priced on. ``mlp_input`` is the
    argument to ``_mlp`` — the ``post_attention_layernorm`` output, one matmul
    upstream, which is what the substitution run needs as its starting point.

    Both are captured by observing the shipped code rather than by re-deriving it.
    Re-deriving is how the previous version of this script came to measure a tensor
    the model never produces: it looked plausible, it ran, and it produced numbers.

    The names are patched on ``slowq.model``, not on ``ops``. ``model.py`` does
    ``from .ops import silu``, so the object the layer calls is
    ``slowq.model.silu``; patching ``ops.silu`` would observe nothing at all.
    """
    import slowq.model as model_module

    model = CausalLMModel.from_directory(model_dir)
    ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)

    silu_inputs: list[np.ndarray] = []
    mlp_inputs: list[np.ndarray] = []
    real_silu = model_module.silu
    real_mlp = model_module.DecoderModel._mlp

    def spy_silu(x: np.ndarray) -> np.ndarray:
        silu_inputs.append(np.array(x, copy=True))
        return real_silu(x)

    def spy_mlp(self: Any, prefix: str, x: np.ndarray) -> np.ndarray:
        mlp_inputs.append(np.array(x, copy=True))
        return real_mlp(self, prefix, x)

    model_module.silu = spy_silu
    model_module.DecoderModel._mlp = spy_mlp
    try:
        model.forward(list(ids), model.new_cache(len(ids) + 1))
    finally:
        model_module.silu = real_silu
        model_module.DecoderModel._mlp = real_mlp

    expected = int(model.config.num_hidden_layers)
    if len(silu_inputs) != expected or len(mlp_inputs) != expected:
        raise AssertionError(
            f"expected {expected} silu/mlp calls (one per layer), observed "
            f"{len(silu_inputs)}/{len(mlp_inputs)} — the patch did not observe the "
            "real path, so the capture is worthless rather than merely incomplete"
        )
    return {
        "model": model,
        "silu_input": silu_inputs[layer],
        "mlp_input": mlp_inputs[layer],
        "layer": layer,
        "num_layers": expected,
        "input_ids": [int(i) for i in ids],
    }


def real_gate_activation(model_dir: Path | str, input_ids: Sequence[int], layer: int = 0) -> np.ndarray:
    """The tensor ``silu`` really receives in ``model_dir``'s ``layer``-th MLP."""
    return capture_layer_zero(model_dir, input_ids, layer)["silu_input"]


# --------------------------------------------------------------------------- #
# substitution: the only comparison that is an attribution
# --------------------------------------------------------------------------- #


def substitution(
    captured: dict[str, Any], repeat: int, warmup: int
) -> dict[str, Any]:
    """Time the real MLP and the real decoder layer with old and new ``silu``.

    One variable changes. Same weights, same input tensor, same process, same
    cache object — the layer writes the same K/V slots on every repeat because
    ``encode``'s ``commit`` is never reached, so no repeat pays an allocation the
    previous one did not.

    This is quantity (3). It is the number that can legitimately be called "what
    the rewrite is worth inside a real layer", and it is much smaller than the
    operator's own speedup, because the operator's speedup is a ratio of two
    isolated calls and the layer contains a great deal that is not sigmoid.
    """
    import slowq.model as model_module

    model = captured["model"]
    layer = int(captured["layer"])
    cfg = model.config
    prefix = f"{cfg.tensor_prefix}.layers.{layer}"
    ids = np.asarray(captured["input_ids"], dtype=np.int64)
    tokens = int(ids.shape[0])

    mlp_input = captured["mlp_input"]
    hidden = np.asarray(model.embed(ids), dtype=np.float32)
    positions = np.arange(tokens, dtype=np.int64)
    cos, sin = ops.rope_cos_sin(positions, model.inv_freq)
    mask = ops.causal_attention_mask(tokens, tokens, 0)
    cache = model.new_cache(tokens)

    real_silu = model_module.silu
    out: dict[str, Any] = {"layer": layer, "tokens": tokens}
    try:
        for label, implementation in (("old", old_silu), ("new", real_silu)):
            model_module.silu = implementation
            mlp = timeit(lambda: model._mlp(prefix, mlp_input), repeat, warmup)
            whole = timeit(
                lambda: model._decoder_layer(
                    layer, hidden, cos, sin, cache, 0, tokens, mask, None
                ),
                repeat,
                warmup,
            )
            out[f"mlp_{label}_seconds"] = mlp["median"]
            out[f"layer_{label}_seconds"] = whole["median"]
            out[f"mlp_{label}"] = mlp
            out[f"layer_{label}"] = whole
    finally:
        model_module.silu = real_silu

    out["mlp_saved_seconds"] = out["mlp_old_seconds"] - out["mlp_new_seconds"]
    out["layer_saved_seconds"] = out["layer_old_seconds"] - out["layer_new_seconds"]
    # A saved-time share, not a speedup: "the rewrite removes X% of a real layer"
    # is the claim that survives, and it is bounded by 100% by construction. The
    # old "153%" had no such bound because it was not a share of anything real.
    out["mlp_saved_share"] = out["mlp_saved_seconds"] / out["mlp_old_seconds"]
    out["layer_saved_share"] = out["layer_saved_seconds"] / out["layer_old_seconds"]
    out["operator_speedup_is_not_this"] = (
        "the operator's own old/new ratio (7.20x on the published run) is a ratio "
        "of two isolated calls; the layer share above is what the rewrite removes "
        "from a real forward. Quoting the operator ratio as a layer or MLP gain "
        "overstates it by however much of the layer is not sigmoid."
    )
    return out


# --------------------------------------------------------------------------- #
# bit equality
# --------------------------------------------------------------------------- #


def equality_cases(captured: np.ndarray, tokens: int, intermediate: int) -> dict[str, np.ndarray]:
    """Every input where the two forms could plausibly diverge.

    The captured activation is first and is the only one that describes this model;
    the rest bound the operator's behaviour on inputs chosen to hit each branch.
    """
    rng = np.random.default_rng(20261023)
    cases: dict[str, np.ndarray] = {
        "captured gate activation": captured,
        "standard normal": rng.standard_normal((tokens, intermediate)).astype(np.float32),
        "all positive": rng.uniform(0.0, 20.0, (tokens, intermediate)).astype(np.float32),
        "all negative": rng.uniform(-20.0, 0.0, (tokens, intermediate)).astype(np.float32),
        # The masked form routes x >= 0 (so -0.0 too) through 1/(1+exp(-x)); the
        # new form's np.where uses the same test, so -0.0 must land the same way.
        "signed zeros": np.zeros((tokens, intermediate), dtype=np.float32) * np.where(
            rng.integers(0, 2, (tokens, intermediate)) == 0, 1.0, -1.0
        ).astype(np.float32),
        "overflow range": np.linspace(-200.0, 200.0, tokens * intermediate).reshape(
            tokens, intermediate
        ).astype(np.float32),
        "infinities and nan": np.full((tokens, intermediate), np.nan, dtype=np.float32),
    }
    cases["infinities and nan"][::3] = np.inf
    cases["infinities and nan"][1::3] = -np.inf
    return cases


def check_bit_equality(cases: dict[str, np.ndarray]) -> tuple[list[str], list[str]]:
    """``(failures, notes)``, comparing raw arrays rather than printed summaries.

    Two separate comparisons, because they answer different questions:

    * ``array_equal(..., equal_nan=True)`` — same values, treating every NaN as
      equal to every other NaN. This is *NaN positions match*, not *NaN bit
      patterns match*, and the distinction is reported rather than blurred.
    * byte comparison over the non-NaN elements — the bitwise claim. An aggregate
      error printed by a log line cannot support it; only the raw arrays can.
    """
    failures: list[str] = []
    notes: list[str] = []
    for label, values in cases.items():
        with np.errstate(all="ignore"):
            left, right = old_sigmoid(values), ops.sigmoid(values)
        if not np.array_equal(left, right, equal_nan=True):
            failures.append(f"{label}: values differ under equal_nan comparison")
            continue
        finite = ~(np.isnan(left) | np.isnan(right))
        if left[finite].tobytes() != right[finite].tobytes():
            differing = int(np.sum(left[finite].view(np.uint32) != right[finite].view(np.uint32)))
            failures.append(
                f"{label}: {differing}/{int(finite.sum())} non-NaN elements differ bit-wise"
            )
        elif np.isnan(left).any():
            nan_bits_differ = bool(
                np.any(left[np.isnan(left)].view(np.uint32) != right[np.isnan(right)].view(np.uint32))
            )
            notes.append(
                f"{label}: NaN 位置一致（{int(np.isnan(left).sum())} 个），非 NaN 元素逐位相同；"
                f"NaN 位模式{'不一致' if nan_bits_differ else '也一致'}"
            )
    return failures, notes


def activation_stats(values: np.ndarray) -> dict[str, Any]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "fraction_negative": float((values < 0).mean()),
        "absmax": float(np.abs(values).max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=128, help="prefill chunk size")
    parser.add_argument("--repeat", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    # Before the two tensor loads below, which are this probe's expensive part on a
    # cold page cache; see out_guard.preflight.
    out_guard.preflight(args.out)

    tokens = args.tokens
    prefix = f"model.layers.{args.layer}"
    intermediate = int(load_tensors(args.model, f"{prefix}.mlp.gate_proj.weight").shape[0])
    # down_proj is [out=hidden, in=intermediate], so hidden is axis 0 — reading it
    # off axis 1 would silently make hidden == intermediate and every "is this the
    # hidden-width input?" test below would take the wrong branch.
    hidden = int(load_tensors(args.model, f"{prefix}.mlp.down_proj.weight").shape[0])

    # Fixed input ids, and the same ones the previous version of this probe used
    # implicitly: it read embedding rows 0..tokens-1, which is token ids
    # 0..tokens-1. Keeping them means the corrected numbers differ from the
    # published ones only in the forward path, not in the input.
    ids = np.arange(tokens, dtype=np.int64)
    vocab = int(load_tensors(args.model, "model.embed_tokens.weight").shape[0])
    if tokens > vocab:
        raise SystemExit(f"--tokens {tokens} exceeds vocab_size {vocab}")

    print(
        f"{args.model.name} layer {args.layer}: hidden={hidden} intermediate={intermediate} "
        f"tokens={tokens} numpy={np.__version__} "
        f"blas={np.show_config('dicts')['Build Dependencies']['blas']['name']}"
    )

    captured = capture_layer_zero(args.model, ids, args.layer)
    real = captured["silu_input"]
    mlp_input = captured["mlp_input"]

    # The comparison that shows what the old probe was pricing. Computed here
    # rather than asserted in prose: the shortcut is cheap to run and its distance
    # from the truth is the size of the error in the published attribution.
    norm_weight = np.asarray(
        load_tensors(args.model, f"{prefix}.input_layernorm.weight"), dtype=np.float32
    )
    shortcut = ops.linear(
        ops.rms_norm(np.asarray(real_embed(args.model, ids)), norm_weight, 1e-6),
        load_tensors(args.model, f"{prefix}.mlp.gate_proj.weight"),
    )
    relative = float(
        np.linalg.norm(real.astype(np.float64) - shortcut.astype(np.float64))
        / max(float(np.linalg.norm(real.astype(np.float64))), 1e-12)
    )

    print("\n[the captured activation vs the shortcut the old probe used]")
    print("  shortcut = rms_norm(embed_tokens(ids), input_layernorm.weight) @ gate_proj.weight.T")
    print(f"  rel_l2(shortcut, captured) = {relative:.4f}")
    for label, values in (("captured", real), ("shortcut", shortcut)):
        stats = activation_stats(values)
        print(
            f"  {label:<9s} mean={stats['mean']:+.4f} std={stats['std']:.4f} "
            f"frac_negative={stats['fraction_negative']:.3f} absmax={stats['absmax']:.3f}"
        )
    print(
        "  注意 RMSNorm 不减均值：上面 captured 的均值不为 0 就是证据，"
        "「出自 RMSNorm 所以天然居中」这句话是错的。"
    )

    failures, notes = check_bit_equality(equality_cases(real, tokens, intermediate))
    print("\n[bit equality]")
    for note in notes:
        print(f"  note {note}")
    if failures:
        for item in failures:
            print(f"  FAIL {item}")
    else:
        print(
            f"  old and new agree on all {len(equality_cases(real, tokens, intermediate))} "
            "input families: NaN positions match and every non-NaN element is "
            "bit-identical, including signed zeros and ±200"
        )

    rng = np.random.default_rng(20261023)
    distributions: dict[str, np.ndarray] = {"captured gate activation": real}
    distributions["standard normal"] = rng.standard_normal((tokens, intermediate)).astype(np.float32)
    distributions["all positive"] = rng.uniform(0.0, 20.0, (tokens, intermediate)).astype(np.float32)
    distributions["all negative"] = rng.uniform(-20.0, 0.0, (tokens, intermediate)).astype(np.float32)

    print(f"\n[{QUANTITIES['operator']}]")
    print(f"  {'input':<26} {'old ms':>9} {'new ms':>9} {'speedup':>9}")
    sigmoid_old_total: dict[str, float] = {}
    sigmoid_rows: dict[str, dict[str, Any]] = {}
    for label, values in distributions.items():
        before = timeit(lambda v=values: old_sigmoid(v), args.repeat, args.warmup)
        after = timeit(lambda v=values: ops.sigmoid(v), args.repeat, args.warmup)
        sigmoid_old_total[label] = before["median"]
        sigmoid_rows[label] = {
            "old_seconds": before["median"],
            "new_seconds": after["median"],
            "speedup": before["median"] / after["median"],
            "old": before,
            "new": after,
            "synthetic": label in SYNTHETIC_DISTRIBUTIONS,
            "purpose": SYNTHETIC_DISTRIBUTIONS.get(
                label, "本模型真实前向里 silu 收到的张量，是唯一能代表本代码库的一行"
            ),
        }
        print(
            f"  {label:<26} {before['median'] * 1e3:>9.3f} {after['median'] * 1e3:>9.3f} "
            f"{before['median'] / after['median']:>8.2f}×"
        )

    activations = rng.standard_normal((tokens, hidden)).astype(np.float32)
    matmuls = [
        ("q_proj", load_tensors(args.model, f"{prefix}.self_attn.q_proj.weight")),
        ("k_proj", load_tensors(args.model, f"{prefix}.self_attn.k_proj.weight")),
        ("v_proj", load_tensors(args.model, f"{prefix}.self_attn.v_proj.weight")),
        ("o_proj", load_tensors(args.model, f"{prefix}.self_attn.o_proj.weight")),
        ("gate_proj", load_tensors(args.model, f"{prefix}.mlp.gate_proj.weight")),
        ("up_proj", load_tensors(args.model, f"{prefix}.mlp.up_proj.weight")),
        ("down_proj", load_tensors(args.model, f"{prefix}.mlp.down_proj.weight")),
    ]
    print(f"\n[{QUANTITIES['matmul_sum']}]")
    print(f"  {'matmul':<12} {'shape':<26} {'ms':>9} {'GFLOP/s':>9}")
    results: dict[str, dict[str, Any]] = {}
    total = 0.0
    for name, weight in matmuls:
        # The input width comes from the weight, not from a hardcoded guess: o_proj
        # takes num_heads * head_dim columns, which for this model is 2048 and not
        # hidden_size. Getting that wrong would time a matmul the layer never does.
        in_features = int(weight.shape[1])
        x = activations if in_features == hidden else rng.standard_normal(
            (tokens, in_features)
        ).astype(np.float32)
        np.asarray(weight)  # touch the pages once; the mmap stays resident
        timed = timeit(lambda x=x, w=weight: ops.linear(x, w), args.repeat, args.warmup)
        flops = 2 * tokens * in_features * int(weight.shape[0])
        total += timed["median"]
        results[name] = {
            "median": timed["median"],
            "min": timed["min"],
            "max": timed["max"],
            "gflops": flops / timed["median"] / 1e9,
        }
        print(
            f"  {name:<12} {f'{tokens}x{in_features} @ {in_features}x{weight.shape[0]}':<26} "
            f"{timed['median'] * 1e3:>9.3f} {flops / timed['median'] / 1e9:>9.1f}"
        )
    print(f"  {'合计':<12} {'':<26} {total * 1e3:>9.3f}")
    print(
        "  这个合计**不是**整层 profile：它没有 softmax、没有 norm、没有残差加、"
        "没有任何逐元素运算，而且每个 matmul 是在热循环里单独计时的。"
    )

    replaced = substitution(captured, args.repeat, args.warmup)
    print(f"\n[{QUANTITIES['real_substitution']}]")
    print(f"  真实 MLP（layer {args.layer}，同一份 post_attention_layernorm 输出）:")
    print(
        f"    old silu {replaced['mlp_old_seconds'] * 1e3:.3f} ms -> "
        f"new silu {replaced['mlp_new_seconds'] * 1e3:.3f} ms, "
        f"省下 {replaced['mlp_saved_seconds'] * 1e3:.3f} ms "
        f"= 真实 MLP 的 {replaced['mlp_saved_share'] * 100:.1f}%"
    )
    print(f"  真实 decoder layer（含 attention / 两个 norm / 两处残差）:")
    print(
        f"    old silu {replaced['layer_old_seconds'] * 1e3:.3f} ms -> "
        f"new silu {replaced['layer_new_seconds'] * 1e3:.3f} ms, "
        f"省下 {replaced['layer_saved_seconds'] * 1e3:.3f} ms "
        f"= 真实整层的 {replaced['layer_saved_share'] * 100:.1f}%"
    )

    print(f"\n[{QUANTITIES['end_to_end']}]")
    print("  本脚本不测端到端。看 artifacts/perf-round1*/ab/ 的 workloads 段。")

    new_sigmoid = sigmoid_rows["captured gate activation"]["new_seconds"]
    verdict = sigmoid_old_total["captured gate activation"] / total
    print("\n[四个量放在一起]")
    print(f"  (1) 算子：captured 输入上 old {sigmoid_old_total['captured gate activation'] * 1e3:.3f} ms"
          f" / new {new_sigmoid * 1e3:.3f} ms = "
          f"{sigmoid_old_total['captured gate activation'] / new_sigmoid:.2f}×")
    print(f"  (2) matmul 之和：{total * 1e3:.3f} ms；(1)/(2) = {verdict * 100:.1f}%"
          " —— 这个比值**不是**任何真实层的占比")
    print(f"  (3) 真实整层替换：省下 {replaced['layer_saved_share'] * 100:.1f}%")
    print(f"  (4) 端到端：见 ab/ workloads，本脚本不提供")

    worst_label = max(sigmoid_old_total, key=lambda k: sigmoid_old_total[k])
    one_sided = [
        sigmoid_rows[label]["speedup"] for label in SYNTHETIC_DISTRIBUTIONS
        if label.startswith("all ")
    ]
    print(
        f"\n  masked 写法的代价取决于输入跨不跨 0：全正/全负只有 "
        f"{min(one_sided):.2f}–{max(one_sided):.2f}×，"
        f"四种输入里最贵的是 {worst_label}。"
    )

    if args.out:
        # Refuses an existing path rather than truncating it; see out_guard.py.
        out = out_guard.write_text(
            args.out,
            json.dumps(
                {
                    "quantities": QUANTITIES,
                    "model": args.model.name,
                    "model_dir": str(args.model),
                    "layer": args.layer,
                    "tokens": tokens,
                    "hidden": hidden,
                    "intermediate": intermediate,
                    "input_ids": captured["input_ids"],
                    "numpy": np.__version__,
                    "blas": np.show_config("dicts")["Build Dependencies"]["blas"]["name"],
                    # Narrower than the A/B harnesses' source_identity block, and
                    # deliberately so: this probe prices one function, so it names
                    # the file that function lives in plus itself. source_identity
                    # would name the whole package and the whole tools/perf
                    # directory, which is more provenance than this number needs and
                    # would make two of its fields change whenever an unrelated
                    # harness is edited.
                    #
                    # Two hashes per file, not one; see code_sha256's docstring for
                    # why the docstring-insensitive one is what keeps this
                    # self-referential — the numbers land in ops.py's own prose.
                    "identity": {
                        "ops_py_sha256": sha256_file(ROOT / "src" / "slowq" / "ops.py"),
                        "model_py_sha256": sha256_file(ROOT / "src" / "slowq" / "model.py"),
                        "probe_sha256": sha256_file(Path(__file__).resolve()),
                        "ops_py_code_sha256": code_sha256(ROOT / "src" / "slowq" / "ops.py"),
                        "model_py_code_sha256": code_sha256(ROOT / "src" / "slowq" / "model.py"),
                        "probe_code_sha256": code_sha256(Path(__file__).resolve()),
                    },
                    "bit_equality_failures": failures,
                    "bit_equality_notes": notes,
                    "gate_activation": activation_stats(real),
                    "mlp_input": activation_stats(mlp_input),
                    "shortcut_vs_captured": {
                        "shortcut": "rms_norm(embed_tokens(ids), input_layernorm.weight) @ gate_proj.weight.T",
                        "rel_l2": relative,
                        "shortcut_stats": activation_stats(shortcut),
                        "why_it_is_wrong": (
                            "跳过了整个 attention 块、第一处残差加，并且用了 "
                            "input_layernorm 而不是 post_attention_layernorm"
                        ),
                    },
                    "sigmoid_old_seconds": sigmoid_old_total,
                    "sigmoid_new_seconds": new_sigmoid,
                    "sigmoid_rows": sigmoid_rows,
                    "matmuls": results,
                    "matmul_total_seconds": total,
                    "matmul_total_is_not_a_layer_profile": True,
                    # Kept under their published names so the old report stays
                    # reproducible from this artifact; both are quantity (1)/(2),
                    # which is not a share of any real layer.
                    "old_sigmoid_share_of_matmuls": verdict,
                    "new_sigmoid_share_of_matmuls": new_sigmoid / total,
                    "sigmoid_speedup_real_gate": (
                        sigmoid_old_total["captured gate activation"] / new_sigmoid
                    ),
                    "real_substitution": replaced,
                    "correction": {
                        "withdrawn": {
                            "old_share_of_matmuls_as_layer_share": (
                                "153% 是 (1)/(2)，即孤立算子除以 7 个热 matmul 中位数之和。"
                                "它不是任何真实层的占比，不能再叫「sigmoid 占整层 matmul 预算的 153%」"
                            ),
                            "operator_speedup_as_mlp_gain": (
                                "7.20× 是算子自身的新旧比值，不是 MLP 或整层变快了 7.20×"
                            ),
                            "negative_fraction_as_real": (
                                "38.8% 为负是**错误激活**上的统计；真实捕获值见 gate_activation"
                            ),
                            "centred_by_construction": (
                                "「出自 RMSNorm 所以天然居中」是错的：RMSNorm 除以均方根，"
                                "从不减均值。LayerNorm 才减均值"
                            ),
                        },
                        "replaced_by": "real_substitution 段（真实 MLP 与真实整层，只替换 silu）",
                        "preserved": "旧数字全部保留在上面，未删除、未改写",
                    },
                },
                ensure_ascii=False,
                indent=1,
            ),
        )
        print(f"\nwrote {out_guard.display(out)}")
    return 1 if failures else 0


def real_embed(model_dir: Path, ids: np.ndarray) -> np.ndarray:
    """Embedding rows for ``ids``, read straight off the mmap'd table."""
    table = np.asarray(load_tensors(model_dir, "model.embed_tokens.weight"), dtype=np.float32)
    return table[ids]


if __name__ == "__main__":
    raise SystemExit(main())
