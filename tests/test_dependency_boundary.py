"""The runtime dependency boundary: numpy + tokenizers + Jinja2, nothing else.

torch, transformers, safetensors and huggingface_hub are allowed only as
conversion/validation *tool* dependencies (the ``slowq[ref]`` extra). Keeping
them out of the runtime import graph is what makes the clean-environment check
in the spec possible, and it is easy to break by accident — one convenience
import in a helper is enough. Two guards:

* a subprocess imports every runtime module and reports which forbidden modules
  ended up in ``sys.modules`` (this catches lazy and transitive imports, and it
  is meaningful because the development venv *does* have torch installed);
* a source scan, which additionally catches an import sitting inside a function
  that no test happens to call.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_MODULES = (
    "slowq",
    "slowq.cache",
    "slowq.cli",
    "slowq.config",
    "slowq.embedding",
    "slowq.engine",
    "slowq.errors",
    "slowq.metrics",
    "slowq.model",
    "slowq.ops",
    "slowq.sampling",
    "slowq.tokenizer",
    "slowq.vlm",
    "slowq.weights",
)

#: Module names the runtime must never pull in, and the top-level packages that
#: would show up in ``sys.modules`` if it did.
FORBIDDEN = (
    "torch",
    "transformers",
    "safetensors",
    "huggingface_hub",
    "onnxruntime",
    "llama_cpp",
    "sentencepiece",
    "accelerate",
)

#: Pillow is an *optional* dependency (``slowq[vision]``). Importing the vision
#: module must not import it — only actually reading an image may — or a
#: text-only install would break on ``import slowq.vlm``.
OPTIONAL = ("PIL",)

PROBE = """
import sys
for name in {modules!r}:
    __import__(name)
leaked = sorted(
    top
    for top in {forbidden!r}
    if top in sys.modules or any(m.startswith(top + ".") for m in sys.modules)
)
sys.stdout.write(",".join(leaked))
"""


def test_importing_the_runtime_does_not_pull_in_a_deep_learning_framework():
    """Run in a subprocess so this test's own imports cannot mask a leak."""
    probe = subprocess.run(
        [sys.executable, "-c", PROBE.format(modules=RUNTIME_MODULES, forbidden=FORBIDDEN)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout == "", f"the runtime imported: {probe.stdout.split(',')}"


def test_importing_the_vision_module_does_not_require_pillow():
    """A text-only environment must still be able to import every runtime module."""
    probe = subprocess.run(
        [sys.executable, "-c", PROBE.format(modules=RUNTIME_MODULES, forbidden=OPTIONAL)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout == "", (
        f"importing the runtime pulled in {probe.stdout.split(',')}, which is an "
        "optional dependency; slowq.vlm must import it inside the function that "
        "reads an image"
    )


def test_no_runtime_source_file_even_mentions_a_forbidden_import():
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "src" / "slowq").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                if name.split(".")[0] in FORBIDDEN:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} {name}")
    assert offenders == []


def test_the_declared_runtime_dependencies_are_the_three_allowed_ones():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = sorted(
        requirement.split(">=")[0].split("==")[0].split("[")[0].strip().lower()
        for requirement in pyproject["project"]["dependencies"]
    )
    assert declared == ["jinja2", "numpy", "tokenizers"]


def test_the_reference_tools_are_an_optional_extra_not_a_runtime_requirement():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    assert "ref" in extras
    assert any(requirement.startswith("torch") for requirement in extras["ref"])
    assert any(requirement.startswith("transformers") for requirement in extras["ref"])
    # 'dev' must stay test-only; a torch dependency there would leak into CI.
    assert not any(requirement.split(">=")[0] in FORBIDDEN for requirement in extras["dev"])


def test_pillow_is_an_optional_vision_extra_and_never_a_runtime_dependency():
    """A text-only install must not be asked for an image library."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    runtime = [requirement.lower() for requirement in pyproject["project"]["dependencies"]]
    assert not any(requirement.startswith("pillow") for requirement in runtime)
    extras = pyproject["project"]["optional-dependencies"]
    assert "vision" in extras
    assert any(
        requirement.lower().startswith("pillow") for requirement in extras["vision"]
    ), extras["vision"]
