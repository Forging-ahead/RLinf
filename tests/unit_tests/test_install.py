# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Check installer selection without downloading packages or launching a GPU."""

import os
import subprocess
import sys
from pathlib import Path
from shlex import quote

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


def test_behavior_selects_sdk_torch_version():
    result = run_functions(
        'ENV_NAME=behavior; TORCH_VERSION=""; apply_env_default_torch\n'
        'test "$TORCH_VERSION" = 2.7.0\n'
    )
    assert result.returncode == 0, result.stderr


def test_behavior_install_reuses_common_helpers():
    result = run_functions(
        "create_and_sync_venv() { echo CREATE; }\n"
        "install_common_embodied_deps() { echo COMMON; }\n"
        "install_flash_attn() { return 0; }\n"
        "install_behavior_env() { echo SDK393; }\n"
        'uv() { echo "UV $*"; }\n'
        "python() { cat >/dev/null; echo 3.11; }\n"
        "cp() { return 0; }\n"
        "bash() { return 0; }\n"
        "ENV_NAME=behavior\ninstall_openpi_model\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[:2] == ["CREATE", "COMMON"]
    assert "SDK393" in result.stdout
    assert "UV pip install rlinf-openpi==0.1.1" in result.stdout


@pytest.mark.skipif(sys.platform != "linux", reason="BEHAVIOR installer runs on Linux")
@pytest.mark.parametrize(
    ("shared", "ref"), [(False, "release/b1k"), (True, "release/b1k"), (True, "my-pin")]
)
def test_sdk_setup_pins_only_the_floating_dependency_in_the_venv(tmp_path, shared, ref):
    venv = tmp_path / "venv"
    source = tmp_path / "shared" if shared else venv / "BEHAVIOR-1K"
    venv.mkdir()
    source.mkdir(parents=True)
    (source / "bddl3").mkdir()
    (source / "joylo").mkdir()
    package = tmp_path / "omnigibson-dev" if shared else source / "OmniGibson"
    package.mkdir()
    if shared:
        (source / "OmniGibson").symlink_to(package, target_is_directory=True)
    metadata = package / "setup.py"
    dependency = "lerobot[dataset] @ git+https://github.com/wensi-ai/lerobot@"
    metadata.write_text(dependency + ref + "\n")
    setup = source / "setup.sh"
    setup.write_text('#!/bin/bash\ncat OmniGibson/setup.py\necho "setup $*"\n')
    setup.chmod(0o755)
    for _ in range(2):
        result = run_functions(
            f"VENV_DIR={quote(str(venv))}\n"
            f"export BEHAVIOR_PATH={quote(str(source)) if shared else ''}\n"
            'uv() { echo "uv $*"; }\n'
            "MODEL=openpi\ninstall_behavior_env\n"
        )
        assert result.returncode == 0, result.stderr
        expected = (
            "436812bd8ee39b768c645c248c91f1330834e687" if ref == "release/b1k" else ref
        )
        assert dependency + expected in result.stdout
        assert (
            "setup --omnigibson --bddl --joylo --confirm-no-conda --accept-nvidia-eula"
            in result.stdout
        )
        assert "uv pip check" in result.stdout
        if shared:
            assert metadata.read_text() == dependency + ref + "\n"
