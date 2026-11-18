# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real evaluation launcher across its Python command boundary."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("explicit_isaac", [False, True])
def test_evaluation_launcher_quotes_hydra_output_path(tmp_path, explicit_isaac):
    repo = tmp_path / "repo with spaces & behavior"
    evaluations = repo / "evaluations"
    (evaluations / "behavior").mkdir(parents=True)
    (evaluations / "behavior" / "behavior_test.yaml").write_text("{}\n")
    source = Path(__file__).resolve().parents[2] / "evaluations" / "run_eval.sh"
    shutil.copyfile(source, evaluations / "run_eval.sh")
    sim_setup = source.parents[1] / "examples/embodiment/sim_env.sh"
    destination = repo / "examples/embodiment/sim_env.sh"
    destination.parent.mkdir(parents=True)
    shutil.copyfile(sim_setup, destination)
    recorder = tmp_path / "record.py"
    output = tmp_path / "arguments.json"
    recorder.write_text(
        "import json,os,sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['ARGUMENTS_OUTPUT']).write_text(json.dumps({\n"
        "'args':sys.argv[1:], 'isaac':{key:os.environ.get(key) for key in "
        "['ISAAC_PATH','EXP_PATH','CARB_APP_PATH']}}))\n"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text('#!/bin/sh\nexec "$REAL_PYTHON" "$ARGUMENT_RECORDER" "$@"\n')
    python.chmod(0o755)
    env = os.environ.copy()
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}",
        REAL_PYTHON=sys.executable,
        ARGUMENT_RECORDER=str(recorder),
        ARGUMENTS_OUTPUT=str(output),
    )
    for key in ("ISAAC_PATH", "EXP_PATH", "CARB_APP_PATH"):
        env.pop(key, None)
    if explicit_isaac:
        env["ISAAC_PATH"] = str(tmp_path / "standalone-isaac")
    result = subprocess.run(
        [
            "bash",
            str(evaluations / "run_eval.sh"),
            "behavior",
            "behavior_test",
            "rollout.model.model_path=/tmp/public-checkpoint",
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    recorded = json.loads(output.read_text())
    args = recorded["args"]
    log_path = next(arg for arg in args if arg.startswith("runner.logger.log_path="))
    assert log_path.startswith('runner.logger.log_path="')
    assert log_path.endswith('"')
    assert "repo with spaces & behavior/logs/" in log_path
    assert args[0] == str(evaluations / "eval_embodied_agent.py")
    if explicit_isaac:
        assert recorded["isaac"] == {
            "ISAAC_PATH": env["ISAAC_PATH"],
            "EXP_PATH": env["ISAAC_PATH"] + "/apps",
            "CARB_APP_PATH": env["ISAAC_PATH"] + "/kit",
        }
    else:
        assert all(value is None for value in recorded["isaac"].values())
