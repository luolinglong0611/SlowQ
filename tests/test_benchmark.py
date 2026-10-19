"""Regression tests for ``tools/benchmark.py``: the report must match the run.

Two ways this tool can lie, both of which happened:

* it timed a *single* full forward and printed ``chunks: 4`` next to it, so the
  report described a chunked prefill that never executed. The chunk count now
  comes from counting calls to ``model.forward``, and both paths are measured.
* it accepted ``--threads 4`` but not ``--threads=4``, so the second spelling
  exported nothing while the report went on to list the BLAS backend — reading
  like a measurement of a pin that was never applied.

Everything here runs offline on the tiny model from ``conftest.py`` or on a
recording stand-in; no real checkpoint is loaded.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from slowq import Engine

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = REPO_ROOT / "tools" / "benchmark.py"

#: Duplicated on purpose: a test that strips the thread variables out of a child
#: environment must not silently stop stripping them if the tool's list grows.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

#: Runs the benchmark module up to (and including) its imports, then reports what
#: the pre-parse did. ``run_name`` is not ``__main__``, so ``main()`` never runs.
PRE_IMPORT_PROBE = """\
import json, os, runpy, sys

bench = os.environ["SLOWQ_BENCH_PATH"]
sys.argv = ["benchmark.py", *sys.argv[1:]]
module = runpy.run_path(bench, run_name="slowq_benchmark")
print("PROBE " + json.dumps({
    "requested": module["REQUESTED_THREADS"],
    "before_numpy": module["THREAD_LIMIT_APPLIED_BEFORE_NUMPY"],
    "env": {name: os.environ.get(name) for name in module["THREAD_ENV_VARS"]},
}))
"""


def load_benchmark_tool():
    """Import ``tools/benchmark.py`` without making ``tools`` a package."""
    spec = importlib.util.spec_from_file_location("slowq_benchmark", BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return load_benchmark_tool()


def child_env(**overrides: str) -> dict[str, str]:
    """The current environment minus any thread pin, so a child starts unpinned."""
    env = {key: value for key, value in os.environ.items() if key not in THREAD_ENV_VARS}
    env.update(overrides)
    return env


def run_tool(*argv: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(BENCHMARK), *argv],
        capture_output=True,
        text=True,
        timeout=600,
        env=env if env is not None else child_env(),
        cwd=str(REPO_ROOT),
        check=False,
    )


def run_pre_import_probe(*argv: str) -> dict:
    """Execute the tool's pre-import section in a fresh interpreter."""
    proc = subprocess.run(
        [sys.executable, "-c", PRE_IMPORT_PROBE, *argv],
        capture_output=True,
        text=True,
        timeout=300,
        env=child_env(SLOWQ_BENCH_PATH=str(BENCHMARK)),
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    line = next(line for line in proc.stdout.splitlines() if line.startswith("PROBE "))
    return json.loads(line[len("PROBE ") :])


class RecordingModel:
    """Records forward calls instead of doing arithmetic.

    ``time_prefill`` only needs ``new_cache`` and ``forward``, so a stand-in is
    enough to check *what was called* — which is exactly what the report claims.
    """

    class Cache:
        def __init__(self, max_context: int) -> None:
            self.max_context = max_context
            self.length = 0

    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def new_cache(self, max_context: int) -> "RecordingModel.Cache":
        return self.Cache(max_context)

    def forward(self, input_ids, cache, *args, **kwargs):
        self.calls.append(list(input_ids))
        cache.length += len(input_ids)
        return None


# --------------------------------------------------------------------------- #
# --threads: parsing and exporting
# --------------------------------------------------------------------------- #


def test_the_test_and_the_tool_agree_on_which_variables_pin_the_blas(tool):
    assert set(THREAD_ENV_VARS) == set(tool.THREAD_ENV_VARS)


@pytest.mark.parametrize(
    "argv",
    [
        ["--threads", "4"],
        ["--threads=4"],
        ["--model", "somewhere", "--threads", "4", "--repeat", "2"],
        ["--model", "somewhere", "--threads=4", "--repeat", "2"],
    ],
)
def test_both_spellings_of_the_flag_parse_to_the_same_number(tool, argv):
    assert tool.requested_threads(argv) == 4


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--model", "somewhere"],
        ["--threads"],  # no value: argparse reports it, the pre-parse stays out of the way
        ["--threads=abc"],
        ["--threadsmith", "4"],
    ],
)
def test_nothing_is_pinned_without_a_usable_value(tool, argv):
    assert tool.requested_threads(argv) is None


@pytest.mark.parametrize("argv", [["--threads", "4"], ["--threads=4"]])
def test_the_pin_reaches_the_environment_before_numpy_loads(argv):
    """In a fresh interpreter, where NumPy really has not been imported yet.

    ``THREAD_LIMIT_APPLIED_BEFORE_NUMPY`` is the tool's own record of the order;
    a BLAS reads these variables once at library load, so exporting them later
    would change nothing while the report still listed the request.
    """
    probe = run_pre_import_probe(*argv)
    assert probe["requested"] == 4
    assert probe["before_numpy"] is True
    assert probe["env"] == dict.fromkeys(THREAD_ENV_VARS, "4")


def test_a_run_without_the_flag_pins_nothing():
    probe = run_pre_import_probe("--model", "somewhere")
    assert probe["requested"] is None
    assert probe["env"] == dict.fromkeys(THREAD_ENV_VARS, None)


@pytest.mark.parametrize("argv", [["--threads=abc"], ["--threads", "abc"], ["--threads"]])
def test_an_unusable_thread_value_is_argparses_error_not_a_silent_default(argv):
    proc = run_tool(*argv, "--model", "models/does-not-exist")
    assert proc.returncode == 2
    assert "usage:" in proc.stderr
    assert "invalid int value" in proc.stderr or "expected one argument" in proc.stderr


def test_the_thread_report_keeps_the_request_apart_from_the_detection(tool, monkeypatch):
    """The requested pin is a setting; only the backend is detected. Neither is
    a measurement of the thread count NumPy ran with, so that field is null
    rather than filled in from the request."""
    monkeypatch.setattr(tool, "REQUESTED_THREADS", 4)
    monkeypatch.setattr(tool, "THREAD_LIMIT_APPLIED_BEFORE_NUMPY", True)
    report = tool.thread_report()
    assert report["requested"] == 4
    assert report["requested_before_numpy_import"] is True
    assert set(report["exported_env"]) == set(THREAD_ENV_VARS)
    assert report["detected_backend"] == tool.blas_info()
    assert report["logical_cpus"] == os.cpu_count()
    assert report["effective_thread_count"] is None
    assert "not observable" in report["note"]


def test_a_run_without_the_flag_reports_no_requested_pin(tool, monkeypatch):
    monkeypatch.setattr(tool, "REQUESTED_THREADS", None)
    report = tool.thread_report()
    assert report["requested"] is None
    assert report["effective_thread_count"] is None


# --------------------------------------------------------------------------- #
# the prefill sweep
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("vocab_size", [2, 3, 372, 151_936])
def test_probe_token_ids_stay_inside_the_vocabulary(tool, vocab_size):
    ids = tool.probe_token_ids(64, vocab_size)
    assert len(ids) == 64
    assert all(0 <= token_id < vocab_size for token_id in ids)
    assert ids == tool.probe_token_ids(64, vocab_size)  # deterministic


def test_the_full_row_is_one_forward_over_the_whole_block(tool):
    model = RecordingModel()
    row = tool.time_prefill(model, list(range(9)), None, mode="full")
    assert row["mode"] == "full"
    assert row["chunk_size"] is None
    assert row["forwards"] == 1 == len(model.calls)
    assert row["forward_token_counts"] == [9]
    assert row["cache_length"] == 9


def test_the_chunked_row_runs_every_chunk_on_one_growing_cache(tool):
    """The chunk count is the number of calls, and the cache keeps growing.

    Nine tokens at a chunk size of four is three forwards of 4/4/1 — an even
    split would hide a loop that stopped early, and a fresh cache per chunk
    would hide a loop that never appended.
    """
    model = RecordingModel()
    row = tool.time_prefill(model, list(range(9)), 4, mode="chunked")
    assert row["mode"] == "chunked"
    assert row["chunk_size"] == 4
    assert row["forwards"] == 3 == len(model.calls)
    assert row["forward_token_counts"] == [4, 4, 1]
    assert [len(call) for call in model.calls] == [4, 4, 1]
    assert row["cache_length"] == 9


def test_the_sweep_measures_both_paths_and_the_calls_are_cross_checked(
    tool, tiny_model_dir: Path, monkeypatch
):
    engine = Engine.from_directory(tiny_model_dir, max_context=64, prefill_chunk_size=3)
    monkeypatch.setattr(tool, "PREFILL_LENGTHS", (9,))

    # An independent counter underneath the tool's own spy: if the report ever
    # stopped reflecting real calls, these two lists would disagree.
    seen: list[int] = []
    real_forward = engine.model.forward

    def counting_forward(input_ids, cache, *args, **kwargs):
        seen.append(len(input_ids))
        return real_forward(input_ids, cache, *args, **kwargs)

    engine.model.forward = counting_forward
    rows = tool.measure_prefill_sweep(engine, max_length=64)

    assert [row["mode"] for row in rows] == ["full", "chunked"]
    full, chunked = rows
    assert (full["forwards"], full["forward_token_counts"]) == (1, [9])
    assert (chunked["forwards"], chunked["forward_token_counts"]) == (3, [3, 3, 3])
    assert chunked["chunk_size"] == 3
    assert full["cache_length"] == chunked["cache_length"] == 9
    assert seen == [9, 3, 3, 3]
    # The derived ``chunks`` field is what made the old report wrong; it is gone.
    assert all("chunks" not in row for row in rows)
    assert all(row["prefill_seconds"] > 0 for row in rows)
    assert engine.model.forward is counting_forward  # the spy restored it


def test_a_length_that_does_not_fit_the_budget_is_skipped_not_padded(
    tool, tiny_model_dir: Path, monkeypatch
):
    engine = Engine.from_directory(tiny_model_dir, max_context=64, prefill_chunk_size=4)
    monkeypatch.setattr(tool, "PREFILL_LENGTHS", (8, 32))
    rows = tool.measure_prefill_sweep(engine, max_length=8)
    assert [row["prompt_tokens"] for row in rows] == [8, 8]
    assert [row["mode"] for row in rows] == ["full", "chunked"]


# --------------------------------------------------------------------------- #
# end to end on the tiny model
# --------------------------------------------------------------------------- #


def test_a_whole_run_reports_only_prefills_it_executed(tiny_model_dir: Path, tmp_path: Path):
    """One real invocation, checked through the JSON it writes and the report it prints."""
    output = tmp_path / "benchmark.json"
    proc = run_tool(
        "--model",
        str(tiny_model_dir),
        "--max-context",
        "64",
        "--prefill-chunk-size",
        "3",
        "--decode-steps",
        "4",
        "--repeat",
        "1",
        "--prompt",
        "你好",
        "--max-new-tokens",
        "4",
        "--threads=2",
        "--json",
        str(output),
    )
    assert proc.returncode == 0, proc.stderr

    payload = json.loads(output.read_text(encoding="utf-8"))
    rows = payload["micro"]["prefill_sweep"]
    # budget = 64 - 4 - 1 = 59, so only the 8- and 32-token probes fit.
    assert [(row["mode"], row["prompt_tokens"], row["forwards"]) for row in rows] == [
        ("full", 8, 1),
        ("chunked", 8, 3),
        ("full", 32, 1),
        ("chunked", 32, 11),
    ]
    for row in rows:
        assert sum(row["forward_token_counts"]) == row["prompt_tokens"]
        assert row["cache_length"] == row["prompt_tokens"]  # prefill only, no decode step
        assert "chunks" not in row
    assert payload["micro"]["cold_first_forward_seconds"] == rows[0]["prefill_seconds"]

    threads = payload["threads"]
    assert threads["requested"] == 2
    assert threads["requested_before_numpy_import"] is True
    assert set(threads["exported_env"].values()) == {"2"}
    assert threads["effective_thread_count"] is None
    assert payload["args"]["threads"] == "2"

    # The printed report has to say the same thing, in the same separation.
    assert "[threads]" in proc.stdout
    assert "requested (--threads)" in proc.stdout
    assert "not observable" in proc.stdout
    assert "chunked@3" in proc.stdout
    assert "forwards" in proc.stdout
