"""Build current sources and smoke-test an installed wheel with/without vision.

Run ``python tools/ci/check_wheel.py`` from any working directory. Only pip and
the standard library are needed by this driver. Package installation may access
the configured package index; no models or historical artifacts are required.
All builds, virtual environments, and probe files live in a temporary directory.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import pkgutil
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
import venv
import zipfile


FORBIDDEN_IMPORTS = (
    "torch", "transformers", "safetensors", "huggingface_hub",
    "onnxruntime", "llama_cpp", "sentencepiece", "accelerate",
)


def clean_environment() -> dict[str, str]:
    env = os.environ.copy()
    # A developer's editable install / model configuration must not affect smoke
    # checks. -I additionally removes Python's cwd and user-site import paths.
    for name in tuple(env):
        if name.startswith("PYTHON") or name.startswith("SLOWQ_MODEL_DIR"):
            env.pop(name)
    env.update(PYTHONUTF8="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


def run(*args: str | Path, cwd: Path) -> None:
    subprocess.run(
        [str(arg) for arg in args], cwd=cwd, env=clean_environment(), check=True,
    )


def probe_installed(expected_version: str, vision: bool) -> None:
    """Executed by the new venv's Python, never by the source-test interpreter."""
    import slowq
    from slowq.errors import UnsupportedInputError
    from slowq.vlm import ImagePreprocessor

    installed_package = Path(sysconfig.get_path("purelib")) / "slowq"
    assert Path(slowq.__file__).resolve().is_relative_to(installed_package.resolve()), (
        f"not importing the installed wheel: {slowq.__file__}"
    )
    assert importlib.metadata.version("slowq") == slowq.__version__ == expected_version

    modules = list(pkgutil.walk_packages(slowq.__path__, "slowq."))
    for module in modules:
        if module.name == "slowq.__main__":
            continue  # The module entry point is checked in its own subprocess.
        imported = importlib.import_module(module.name)
        assert Path(imported.__file__).resolve().is_relative_to(installed_package.resolve())
    leaked = sorted(set(FORBIDDEN_IMPORTS + ("PIL",)) & set(sys.modules))
    assert not leaked, f"runtime import crossed the dependency boundary: {leaked}"
    for name in ("torch", "transformers"):
        assert importlib.util.find_spec(name) is None, f"unexpected dependency: {name}"

    # tokenizers may bring huggingface-hub as a transitive *installed* dependency;
    # the boundary above forbids importing it, not satisfying tokenizers metadata.
    requirements = importlib.metadata.requires("slowq") or []
    direct = {
        re.match(r"[A-Za-z0-9_.-]+", req).group().lower()
        for req in requirements if ";" not in req
    }
    assert direct == {"numpy", "tokenizers", "jinja2"}, direct
    assert (importlib.util.find_spec("PIL") is not None) == vision
    if vision:
        from PIL import Image

        path = Path.cwd() / "smoke.png"
        Image.new("RGB", (2, 3), (17, 29, 41)).save(path)
        loaded = ImagePreprocessor.load(path)
        assert loaded.size == (2, 3) and loaded.getpixel((0, 0)) == (17, 29, 41)
    else:
        try:
            ImagePreprocessor.load(Path.cwd() / "absent.png")
        except UnsupportedInputError as exc:
            assert "slowq[vision]" in str(exc), str(exc)
        else:
            raise AssertionError("reading an image without Pillow should explain the vision extra")

    for entrypoint in (
        [sys.executable, "-I", "-m", "slowq"],
        [str(Path(sys.executable).parent / "slowq")],
    ):
        version = subprocess.check_output(
            [*entrypoint, "--version"], env=clean_environment(), text=True,
        ).strip()
        assert version == f"slowq {expected_version}", version
        for command in ([], ["generate"], ["chat"], ["embed"], ["vlm"], ["inspect"]):
            result = subprocess.run(
                [*entrypoint, *command, "--help"], env=clean_environment(),
                check=True, capture_output=True, text=True,
            )
            assert "usage:" in result.stdout, result.stdout
    print(f"PASS: installed slowq {expected_version}, vision={vision}, {slowq.__file__}")


def check_wheel() -> None:
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    expected_version = project["project"]["version"]
    with tempfile.TemporaryDirectory(prefix="slowq-wheel-") as temporary:
        work = Path(temporary)
        source = work / "source"
        source.mkdir()
        for name in ("pyproject.toml", "README.md", "LICENSE"):
            shutil.copy2(root / name, source / name)
        shutil.copytree(
            root / "src" / "slowq", source / "src" / "slowq",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        # Building a fresh staging tree cannot reuse checkout/build, egg-info, or
        # an old wheel. The checkout and any user build products remain untouched.
        wheels = work / "wheels"
        run(sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-cache-dir",
            "--wheel-dir", wheels, source, cwd=work)
        built = list(wheels.glob("*.whl"))
        assert len(built) == 1, built
        wheel = built[0]
        with zipfile.ZipFile(wheel) as archive:
            packaged = {
                name: archive.read(name) for name in archive.namelist()
                if not name.endswith("/") and ".dist-info/" not in name
            }
        current = {
            path.relative_to(source / "src").as_posix(): path.read_bytes()
            for path in (source / "src" / "slowq").rglob("*") if path.is_file()
        }
        assert packaged == current, "wheel files differ from the staged package sources"

        # Use a cwd with no source package, copied probe, and isolated Python;
        # neither editable .pth files nor PYTHONPATH can rescue a broken wheel.
        probe = work / "check_wheel.py"
        shutil.copy2(Path(__file__), probe)
        for vision in (False, True):
            prefix = work / ("vision-venv" if vision else "text-venv")
            venv.EnvBuilder(with_pip=True).create(prefix)
            python = prefix / "bin" / "python"
            requirement = str(wheel) + ("[vision]" if vision else "")
            run(python, "-m", "pip", "install", "--disable-pip-version-check",
                "--no-input", requirement, cwd=work)
            run(python, "-m", "pip", "check", cwd=work)
            flags = ["--vision"] if vision else []
            run(python, "-I", probe, "--probe-version", expected_version, *flags, cwd=work)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-version", help=argparse.SUPPRESS)
    parser.add_argument("--vision", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe_version:
        probe_installed(args.probe_version, args.vision)
    else:
        check_wheel()
