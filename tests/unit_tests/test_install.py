# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Check installer selection without downloading packages or launching a GPU."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

INSTALLER = Path(__file__).resolve().parents[2] / "requirements" / "install.sh"


def run_functions(body: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    return subprocess.run(
        ["bash", "-c", 'source "$1"\n' + body, "installer-check", str(INSTALLER)],
        env=env,
        text=True,
        capture_output=True,
    )


@pytest.mark.parametrize(
    "python,torch,platform",
    [("3.10", "", "nvidia"), ("3.11.14", "2.8.0", "nvidia"), ("3.11.14", "", "amd")],
)
def test_incompatible_behavior_stack_fails_before_install(python, torch, platform):
    result = run_functions(
        f'ENV_NAME=behavior; PYTHON_VERSION={python}; TORCH_VERSION="{torch}"; PLATFORM={platform}\n'
        "uname() { echo Linux; }\nvalidate_behavior_install\n"
    )
    assert result.returncode != 0
    assert "BEHAVIOR 3.9.3 requires" in result.stderr


def test_behavior_and_model_select_their_respective_torch_versions():
    result = run_functions(
        'ENV_NAME=behavior; TORCH_VERSION=""; apply_env_default_torch\n'
        'test "$TORCH_VERSION" = 2.7.0\n'
        'TARGET=embodied; MODEL=openpi; ENV_NAME=""; TORCH_VERSION=""; apply_model_default_torch\n'
        'test "$TORCH_VERSION" = 2.7.1\n'
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("model", ["", "openpi"])
def test_behavior_install_reuses_common_helpers(model):
    result = run_functions(
        "create_and_sync_venv() { echo CREATE; }\n"
        "install_common_embodied_deps() { echo COMMON; }\n"
        "install_behavior_env() { echo SDK393; }\n"
        'uv() { echo "UV $*"; }\n'
        "python() { cat >/dev/null; echo 3.11; }\n"
        "cp() { return 0; }\n"
        "bash() { return 0; }\n"
        + (
            'ENV_NAME=""\ninstall_openpi_model\n'
            if model
            else "ENV_NAME=behavior\ninstall_env_only\n"
        )
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[:2] == ["CREATE", "COMMON"]
    if model:
        assert "SDK393" not in result.stdout
        assert "UV pip install rlinf-openpi==0.1.1" in result.stdout
    else:
        assert result.stdout.splitlines()[2] == "SDK393"


def test_wrong_reused_sdk_checkout_does_not_run_setup(tmp_path):
    marker = tmp_path / "setup-ran"
    result = run_functions(
        f'clone_or_reuse_repo() {{ echo "{tmp_path}"; }}\n'
        "git() { echo wrong-revision; }\n"
        f'bash() {{ touch "{marker}"; }}\n'
        "install_behavior_env\n"
    )
    assert result.returncode != 0
    assert "BEHAVIOR_PATH must point to" in result.stderr
    assert not marker.exists()


def test_combined_install_routes_model_and_sdk_to_sibling_environments(tmp_path):
    model = tmp_path / "model"
    sdk = tmp_path / "model-behavior"
    for path in (model, sdk):
        (path / "bin").mkdir(parents=True)
        py = path / "bin" / "python"
        py.write_text(
            '#!/bin/sh\nif [ "$1" = -V ]; then echo 3.11.14; else echo 2.52.1; fi\n'
        )
        py.chmod(0o755)
    (model / "bin" / "activate").touch()
    uv = model / "bin" / "uv"
    uv.write_text("#!/bin/sh\nexit 0\n")
    uv.chmod(0o755)
    result = run_functions(
        f'MODEL=openpi; VENV_DIR="{model}"; '
        'bash() { echo "INSTALL $*"; }\n'
        "install_behavior_with_model embodied --model openpi --env behavior --no-root\n"
    )
    assert result.returncode == 0, result.stderr
    calls = result.stdout.splitlines()
    assert len(calls) == 2
    assert f"--no-root --venv {sdk} --env behavior" in calls[0]
    assert f"--no-root --venv {model} --model openpi" in calls[1]
    assert (model / "bin" / "activate").read_text().strip() == (
        f"export BEHAVIOR_ENV_PYTHON={sdk}/bin/python"
    )


@pytest.mark.parametrize("checkout", ["separate", "same", "symlink"])
def test_sdk_dependency_pin_preserves_supplied_source(tmp_path, checkout):
    same_checkout = checkout != "separate"
    original = tmp_path / "supplied"
    build = original if checkout == "same" else tmp_path / "installation"
    dependency = "git+https://github.com/wensi-ai/lerobot@release/b1k"
    for directory in {original} if same_checkout else {original, build}:
        (directory / "OmniGibson").mkdir(parents=True)
        (directory / "OmniGibson/setup.py").write_text(dependency)
        (directory / "setup.sh").write_text("exit 42\n")
    if checkout == "symlink":
        build.symlink_to(original, target_is_directory=True)
    body = (
        f'clone_or_reuse_repo() {{ if [ "$1" = BEHAVIOR_PATH ]; then echo "{original}"; else echo "{build}"; fi; }}\n'
        "git() { echo a8247a8cc1633fe1ca0cc66aa07243d46c64f155; }\n"
        "uv() { return 0; }\n"
        'python() { if [ "$1" = "-c" ]; then return 1; fi; command python "$@"; }\n'
        "install_behavior_env\n"
    )
    for _ in range(2):
        result = run_functions(body)
        assert result.returncode == (1 if same_checkout else 42), result.stderr
        assert (original / "OmniGibson/setup.py").read_text() == dependency
    if not same_checkout:
        assert (
            "436812bd8ee39b768c645c248c91f1330834e687"
            in (build / "OmniGibson/setup.py").read_text()
        )


@pytest.mark.skipif(
    sys.platform != "linux", reason="Native SDK installer requires Linux"
)
def test_sdk_runtime_reuse_keeps_the_complete_runtime(tmp_path):
    sdk = tmp_path / "runtime"
    app = sdk / "isaacsim/apps/complete-runtime.kit"
    app.parent.mkdir(parents=True)
    app.write_text("complete runtime")
    (app.parent.parent / "__init__.py").write_text("")
    for name, version, requirements in [
        ("isaacsim", "5.1.0.0", []),
        ("isaacsim-core", "5.1.0.0", ["isaacsim-kernel==5.1.0.0", "filelock==3.13.1"]),
        ("isaacsim-kernel", "5.1.0.0", ["attrs==25.1.0"]),
        ("torch", "2.7.0", []),
        ("omnigibson", "3.9.3", []),
    ]:
        dist = sdk / f"{name.replace('-', '_')}-{version}.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text(
            f"Name: {name}\nVersion: {version}\n"
            + "".join(f"Requires-Dist: {req}\n" for req in requirements)
        )
    source, build = tmp_path / "supplied", tmp_path / "installation"
    for directory in (source, build):
        (directory / "OmniGibson").mkdir(parents=True)
        (directory / "OmniGibson/setup.py").write_text(
            "git+https://github.com/wensi-ai/lerobot@release/b1k"
        )
        (directory / "setup.sh").write_text("exit 42\n")
    args = tmp_path / "uv-args"
    result = run_functions(
        f'export PYTHONPATH="{sdk}"\n'
        f'clone_or_reuse_repo() {{ if [ "$1" = BEHAVIOR_PATH ]; then echo "{source}"; else echo "{build}"; fi; }}\n'
        "git() { echo a8247a8cc1633fe1ca0cc66aa07243d46c64f155; }\n"
        f'uv() {{ printf "%s\\n" "$@" >> "{args}"; }}\n'
        "install_behavior_env\n"
    )
    assert result.returncode == 0, result.stderr
    requirements = args.read_text().splitlines()
    assert "filelock==3.13.1" in requirements
    assert "attrs==25.1.0" in requirements
    assert not any(req.startswith("isaacsim-") for req in requirements)
    assert app.read_text() == "complete runtime"
