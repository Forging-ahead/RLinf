# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BEHAVIOR environment contracts used by RLinf workers and wrappers."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs.sim.behavior import behavior_env
from rlinf.envs.sim.behavior.instance_loader import ActivityInstanceLoader
from rlinf.envs.sim.behavior.utils import R1PRO_PROPRIO_KEYS, setup_omni_cfg


class FakeBehaviorPool:
    """Supply simulator transitions without Isaac Sim or subprocesses."""

    instances: list[FakeBehaviorPool] = []

    def __init__(self, cfg, num_envs, seed):
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed = seed
        self.activity_name = OmegaConf.select(
            cfg, "omni_config.task.activity_name", default="turning_on_radio"
        )
        self.steps = torch.zeros(num_envs, dtype=torch.int64)
        self.reset_calls = []
        self.actions = []
        self.close_count = 0
        self.terminate_at = {}
        self.truncate_at = {}
        self.instances.append(self)

    @classmethod
    def acquire_shared(cls, cfg, worker_info, stages, num_envs, seed_offset):
        assert stages == 1
        return cls(cfg, num_envs, cfg.seed + seed_offset * num_envs), 0

    def release_shared(self, offset):
        self.close()

    def env_reset_slice(self, start, num_envs, indices):
        return self.reset(indices)

    def env_chunk_step_slice(self, start, num_envs, actions):
        return self.chunk_step(actions)

    def _observation(self, env_id):
        # Different pixel values make head/left/right swaps observable.
        frame = int(self.steps[env_id])
        return {
            "robot_r1": {
                "robot_r1/zed_link:Camera:0": {
                    "rgb": torch.full((4, 5, 3), 10 + frame, dtype=torch.uint8)
                },
                "robot_r1/left_realsense_link:Camera:0": {
                    "rgb": torch.full((4, 5, 3), 20 + frame, dtype=torch.uint8)
                },
                "robot_r1/right_realsense_link:Camera:0": {
                    "rgb": torch.full((4, 5, 3), 30 + frame, dtype=torch.uint8)
                },
                "proprio": torch.arange(256, dtype=torch.float32)
                + env_id * 1000
                + frame * 10000,
            }
        }

    def _info(self, env_id, terminated=False, truncated=False):
        return {
            "episode_length": int(self.steps[env_id]),
            "success": bool(terminated),
            "done": {
                "success": bool(terminated),
                "termination_conditions": {
                    "success": {"done": bool(terminated)},
                    "timeout": {"done": bool(truncated)},
                },
            },
        }

    def reset(self, env_indices=None):
        indices = (
            list(range(self.num_envs))
            if env_indices is None
            else [int(index) for index in env_indices]
        )
        self.reset_calls.append(indices)
        self.steps[indices] = 0
        return (
            [self._observation(index) for index in indices],
            [self._info(index) for index in indices],
        )

    def chunk_step(self, actions):
        actions = torch.as_tensor(actions)
        self.actions.append(actions.clone())
        outputs = ([], [], [], [], [])
        for step in range(actions.shape[1]):
            self.steps += 1
            terminated = torch.tensor(
                [
                    int(self.steps[i]) >= self.terminate_at.get(i, float("inf"))
                    for i in range(self.num_envs)
                ]
            )
            truncated = torch.tensor(
                [
                    int(self.steps[i]) >= self.truncate_at.get(i, float("inf"))
                    for i in range(self.num_envs)
                ]
            )
            skip_obs = self.cfg.get("skip_intermediate_obs_in_chunk", False)
            observations = (
                None
                if skip_obs and step < actions.shape[1] - 1
                else [self._observation(i) for i in range(self.num_envs)]
            )
            outputs[0].append(observations)
            outputs[1].append(torch.arange(1, self.num_envs + 1).float() / 4)
            outputs[2].append(terminated)
            outputs[3].append(truncated)
            outputs[4].append(
                [
                    self._info(i, terminated[i], truncated[i])
                    for i in range(self.num_envs)
                ]
            )
        return outputs

    def close(self):
        self.close_count += 1


@pytest.fixture
def inline_behavior_actors(monkeypatch):
    """Exercise real pool sharding with synchronous process-edge handles."""
    processes = []
    killed = []

    class Process(FakeBehaviorPool):
        def get_activity_name(self):
            return self.activity_name

        def _observation(self, env_id):
            observation = super()._observation(env_id)
            observation["robot_r1"]["proprio"][0] = self.seed + env_id
            return observation

        def chunk_step(self, actions):
            outputs = super().chunk_step(actions)
            for time_index, rows in enumerate(outputs[0]):
                if rows is not None:
                    for local, row in enumerate(rows):
                        row["robot_r1"]["proprio"][1] = actions[local, time_index, 0]
            return outputs

    class ActorFactory:
        def __init__(self, cls):
            self.cls = cls

        def options(self, **kwargs):
            self.actor_options = kwargs
            return self

        def remote(self, *args):
            process = self.cls(*args)
            process.actor_options = self.actor_options
            processes.append(process)
            return SimpleNamespace(
                get_activity_name=SimpleNamespace(remote=process.get_activity_name),
                reset=SimpleNamespace(remote=process.reset),
                chunk_step=SimpleNamespace(remote=process.chunk_step),
            )

    monkeypatch.setattr(behavior_env, "BehaviorProcess", Process)
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(
            remote=lambda **kwargs: ActorFactory,
            get=lambda values: values,
            kill=lambda actor, **kwargs: killed.append(actor),
            get_runtime_context=lambda: SimpleNamespace(
                get_node_id=lambda: "node-test"
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "ray.util.scheduling_strategies",
        SimpleNamespace(NodeAffinitySchedulingStrategy=lambda **kwargs: kwargs),
    )
    return processes, killed


@pytest.fixture
def make_behavior_env(monkeypatch):
    """Keep the real environment adapter and replace its process boundary."""
    FakeBehaviorPool.instances = []
    monkeypatch.setattr(behavior_env, "BehaviorProcessPool", FakeBehaviorPool)
    environments = []

    def create(
        *, num_envs=2, seed_offset=0, stages=1, record_metrics=True, **overrides
    ):
        cfg = OmegaConf.create(
            {
                "device": "cpu",
                "seed": 42,
                "total_num_envs": num_envs * stages,
                "num_env_subprocess": None,
                "auto_reset": False,
                "ignore_terminations": False,
                "use_fixed_reset_state_ids": False,
                "max_episode_steps": 100,
                "reward_coef": 2.0,
                "enable_offload": False,
                "enable_init_offload": True,
                "skip_intermediate_obs_in_chunk": False,
                "base_config_name": "r1pro_behavior",
                "omni_config": {"task": {"activity_name": "turning_on_radio"}},
            }
        )
        cfg.update(overrides)
        env = behavior_env.BehaviorEnv(
            cfg=cfg,
            num_envs=num_envs,
            seed_offset=seed_offset,
            total_num_processes=stages,
            worker_info=SimpleNamespace(group_world_size=1),
            record_metrics=record_metrics,
        )
        environments.append(env)
        return env

    yield create
    for environment in environments:
        environment.close()


def test_behavior_reset_defers_official_score_until_first_step(make_behavior_env):
    env = make_behavior_env(num_envs=1)
    original_info = env.pool._info

    def official_info(env_id, terminated=False, truncated=False):
        return {
            **original_info(env_id, terminated, truncated),
            "q_score": 0.5 if env.pool.steps[env_id] else None,
        }

    env.pool._info = official_info
    _, info = env.reset()
    assert "q_score" not in info["episode"]
    assert info["episode"]["episode_length"].item() == 0
    _, _, _, _, infos = env.chunk_step(torch.zeros(1, 1, 23))
    assert infos[-1]["episode"]["q_score"].item() == 0.5


@pytest.mark.parametrize("task", ["turning_on_radio", "freeze_fruit"])
def test_behavior_reset_preserves_worker_observations(make_behavior_env, task):
    env = make_behavior_env(omni_config={"task": {"activity_name": task}})
    obs, info = env.reset()

    assert set(obs) == {"main_images", "wrist_images", "states", "task_descriptions"}
    assert obs["main_images"].shape == (2, 4, 5, 3)
    assert obs["wrist_images"].shape == (2, 2, 4, 5, 3)
    assert obs["main_images"].dtype == torch.uint8
    assert obs["wrist_images"].dtype == torch.uint8
    assert torch.all(obs["main_images"] == 10)
    assert torch.all(obs["wrist_images"][:, 0] == 20)
    assert torch.all(obs["wrist_images"][:, 1] == 30)
    assert obs["states"].shape == (2, 256)
    torch.testing.assert_close(obs["states"][1], torch.arange(256).float() + 1000)
    assert len(obs["task_descriptions"]) == 2
    assert all(
        isinstance(prompt, str) and prompt for prompt in obs["task_descriptions"]
    )
    assert isinstance(info, dict)
    assert env.seed == 42
    env.is_start = False
    assert env.is_start is False
    env.update_reset_state_ids()


def test_behavior_chunk_preserves_actions_rewards_and_time_axes(make_behavior_env):
    env = make_behavior_env()
    env.reset()
    actions = np.arange(2 * 3 * 23, dtype=np.float32).reshape(2, 3, 23)

    observations, rewards, terminated, truncated, infos = env.chunk_step(actions)

    torch.testing.assert_close(env.pool.actions[-1], torch.from_numpy(actions))
    assert len(observations) == len(infos) == 3
    assert rewards.shape == terminated.shape == truncated.shape == (2, 3)
    assert terminated.dtype == truncated.dtype == torch.bool
    assert not terminated.any() and not truncated.any()
    torch.testing.assert_close(rewards, torch.tensor([[0.5] * 3, [1.0] * 3]))
    assert torch.all(observations[-1]["main_images"] == 13)
    torch.testing.assert_close(infos[-1]["episode"]["return"], torch.tensor([1.5, 3.0]))


@pytest.mark.parametrize("record_metrics", [False, True])
def test_behavior_elapsed_steps_excludes_cached_terminal_transitions(
    make_behavior_env, monkeypatch, record_metrics
):
    env = make_behavior_env(record_metrics=record_metrics)
    env.reset()
    env.pool.terminate_at = {0: 1}
    simulate_chunk = env.pool.chunk_step

    def chunk_with_cached_terminal_state(actions):
        result = simulate_chunk(actions)
        for time_index in range(1, actions.shape[1]):
            result[0][time_index][0] = result[0][0][0]
            result[1][time_index][0] = 0
            result[4][time_index][0] = result[4][0][0]
        env.pool.steps[0] = 1
        return result

    monkeypatch.setattr(env.pool, "chunk_step", chunk_with_cached_terminal_state)
    observations, rewards, terminated, _, infos = env.chunk_step(torch.zeros(2, 4, 23))

    assert terminated[:, -1].tolist() == [True, False]
    assert env.elapsed_steps.tolist() == [1, 4]
    assert torch.all(observations[-1]["main_images"][0] == 11)
    assert torch.all(observations[-1]["main_images"][1] == 14)
    torch.testing.assert_close(rewards[0], torch.tensor([0.5, 0.0, 0.0, 0.0]))
    if record_metrics:
        assert infos[-1]["episode"]["episode_length"].tolist() == [1, 4]
        assert infos[-1]["episode"]["episode_len"].tolist() == [1, 4]


@pytest.mark.parametrize("action_dim", [22, 24])
def test_behavior_rejects_wrong_action_dimension(make_behavior_env, action_dim):
    env = make_behavior_env()
    env.reset()
    with pytest.raises(ValueError, match="23|action"):
        env.chunk_step(torch.zeros(2, 3, action_dim))
    assert env.pool.actions == []


@pytest.mark.parametrize("ignore_terminations", [False, True])
def test_behavior_separates_success_and_time_limit(
    make_behavior_env, ignore_terminations
):
    env = make_behavior_env(ignore_terminations=ignore_terminations)
    env.reset()
    env.pool.terminate_at = {0: 1}
    env.pool.truncate_at = {1: 2}

    _, _, terminated, truncated, _ = env.chunk_step(torch.zeros(2, 3, 23))

    assert not terminated[:, :-1].any()
    assert not truncated[:, :-1].any()
    assert terminated[:, -1].tolist() == [not ignore_terminations, False]
    # Ignoring success must not convert it into a timeout.
    assert truncated[:, -1].tolist() == [False, True]


def test_behavior_auto_reset_preserves_unfinished_env_and_final_transition(
    make_behavior_env,
):
    env = make_behavior_env(auto_reset=True)
    env.reset()
    env.pool.terminate_at = {0: 1}

    observations, _, terminated, truncated, infos = env.chunk_step(
        torch.zeros(2, 2, 23)
    )

    assert env.pool.reset_calls == [[0, 1], [0]]
    assert env.pool.steps.tolist() == [0, 2]
    assert terminated[:, -1].tolist() == [True, False]
    assert not truncated.any()
    final_info = infos[-1]
    assert final_info["_final_observation"].tolist() == [True, False]
    assert final_info["_final_info"].tolist() == [True, False]
    assert torch.all(final_info["final_observation"]["main_images"] == 12)
    assert torch.all(observations[-1]["main_images"][0] == 10)
    assert torch.all(observations[-1]["main_images"][1] == 12)
    torch.testing.assert_close(
        final_info["final_info"]["episode"]["return"], torch.tensor([1.0, 2.0])
    )

    env.pool.terminate_at = {}
    _, _, _, _, next_infos = env.chunk_step(torch.zeros(2, 1, 23))
    torch.testing.assert_close(
        next_infos[-1]["episode"]["return"], torch.tensor([0.5, 3.0])
    )


def test_behavior_keeps_last_observation_when_intermediate_frames_are_skipped(
    make_behavior_env,
):
    env = make_behavior_env(skip_intermediate_obs_in_chunk=True)
    env.reset()
    observations, rewards, _, _, infos = env.chunk_step(torch.zeros(2, 3, 23))

    assert observations[:2] == [None, None]
    assert observations[-1]["states"].shape == (2, 256)
    assert rewards.shape == (2, 3)
    assert len(infos) == 3


def test_behavior_supports_disabled_metrics(make_behavior_env):
    env = make_behavior_env(record_metrics=False)
    observations, info = env.reset()
    results = env.chunk_step(torch.zeros(2, 2, 23))

    assert observations["states"].shape == (2, 256)
    assert info == {}
    assert results[1].shape == (2, 2)
    assert results[-1] == [{}, {}]


def test_behavior_offload_recreates_pool_on_reset_and_close_is_idempotent(
    make_behavior_env,
):
    env = make_behavior_env(enable_offload=True, enable_init_offload=False)
    assert FakeBehaviorPool.instances == []
    env.reset()
    previous_pool = env.pool

    env.offload()
    env.offload()
    assert previous_pool.close_count == 1
    assert env.pool is None
    with pytest.raises(RuntimeError, match="reset"):
        env.chunk_step(torch.zeros(2, 1, 23))
    assert len(FakeBehaviorPool.instances) == 1
    env.reset()
    assert env.pool is not previous_pool
    assert env.pool.seed == previous_pool.seed
    env.close()
    env.close()
    assert FakeBehaviorPool.instances[-1].close_count == 1


def test_behavior_requires_initial_reset_before_actions(make_behavior_env):
    env = make_behavior_env()
    with pytest.raises(RuntimeError, match="reset"):
        env.chunk_step(torch.zeros(2, 1, 23))
    assert env.pool.actions == []


def test_behavior_rejects_auto_reset_with_offload_before_starting_simulator(
    make_behavior_env,
):
    with pytest.raises(ValueError, match="auto_reset|offload"):
        make_behavior_env(auto_reset=True, enable_offload=True)
    assert FakeBehaviorPool.instances == []


def test_behavior_shared_pool_keeps_stages_and_train_eval_independent(
    inline_behavior_actors,
):
    processes, killed = inline_behavior_actors
    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "seed": 42,
            "total_num_envs": 8,
            "num_env_subprocess": 3,
            "ignore_terminations": False,
            "auto_reset": False,
            "max_episode_steps": 100,
            "use_fixed_reset_state_ids": False,
        }
    )
    worker = SimpleNamespace(group_world_size=1)
    # Open stage 1 first, then offload and reattach stage 0 while stage 1 stays active.
    b = behavior_env.BehaviorEnv(cfg, 4, 1, 2, worker)
    a = behavior_env.BehaviorEnv(cfg, 4, 0, 2, worker)
    evaluation = behavior_env.BehaviorEnv(OmegaConf.create(cfg), 4, 0, 2, worker)
    try:
        assert a.pool is b.pool and a.pool is not evaluation.pool
        assert (a.pool_offset, b.pool_offset) == (0, 4)
        assert [list(rows) for rows in a.pool.rows] == [[0, 1], [2, 3], [4, 5, 6, 7]]
        assert len(processes) == 6
        a.reset()
        b.reset()
        a.chunk_step(torch.zeros(4, 3, 23))
        assert [proc.steps.tolist() for proc in processes[:3]] == [
            [3, 3],
            [3, 3],
            [0] * 4,
        ]
        a.offload()
        assert not killed
        b.chunk_step(torch.ones(4, 2, 23))
        a.reset()
        assert b.pool is a.pool
        assert processes[2].steps.tolist() == [2] * 4
        assert len(processes) == 6
        a.close()
        b.close()
        assert len(killed) == 3
        evaluation.reset()
    finally:
        a.close()
        b.close()
        evaluation.close()
    assert len(killed) == 6


@pytest.mark.parametrize("operation", ["step", "reset"])
def test_behavior_partial_shard_failure_requires_full_stage_reset(
    inline_behavior_actors, monkeypatch, operation
):
    processes, _ = inline_behavior_actors
    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "seed": 42,
            "total_num_envs": 4,
            "num_env_subprocess": 4,
            "ignore_terminations": False,
            "auto_reset": False,
            "max_episode_steps": 100,
            "use_fixed_reset_state_ids": False,
        }
    )
    worker = SimpleNamespace(group_world_size=1)
    stage = behavior_env.BehaviorEnv(cfg, 2, 0, 2, worker)
    other = behavior_env.BehaviorEnv(cfg, 2, 1, 2, worker)
    actions = torch.zeros(2, 1, 23)
    try:
        stage.reset()
        other.reset()
        stage.chunk_step(actions)
        second = stage.pool.env_processes[1]

        def fail(*args):
            raise RuntimeError("one shard failed")

        with monkeypatch.context() as patch:
            method = second.chunk_step if operation == "step" else second.reset
            patch.setattr(method, "remote", fail)
            with pytest.raises(RuntimeError, match="one shard failed"):
                if operation == "step":
                    stage.chunk_step(actions)
                else:
                    stage.reset()
        assert processes[0].steps.item() == (2 if operation == "step" else 0)
        assert processes[1].steps.item() == 1
        with pytest.raises(RuntimeError, match="entire BEHAVIOR stage"):
            stage.chunk_step(actions)
        other.chunk_step(actions)
        assert [p.steps.item() for p in processes[2:]] == [1, 1]
        stage.env_reset([0])
        with pytest.raises(RuntimeError, match="entire BEHAVIOR stage"):
            stage.chunk_step(actions)
        stage.reset()
        stage.chunk_step(actions)
        assert [p.steps.item() for p in processes] == [1, 1, 1, 1]
    finally:
        stage.close()
        other.close()


def test_behavior_stages_cannot_split_a_native_physics_clock(inline_behavior_actors):
    processes, _ = inline_behavior_actors
    cfg = OmegaConf.create({"num_env_subprocess": 1, "auto_reset": False})
    with pytest.raises(ValueError, match="2 independent pipeline stages"):
        behavior_env.BehaviorProcessPool(cfg, 2, 42, pipeline_stage_num=2)
    assert not processes


def test_behavior_failure_and_timeout_reset_without_reporting_success(
    make_behavior_env,
):
    env = make_behavior_env(num_envs=3, auto_reset=True)
    env.reset()
    env.pool.terminate_at = {0: 1}
    env.pool.truncate_at = {1: 1}
    info_for_step = env.pool._info

    def with_failure(index, terminated=False, truncated=False):
        info = info_for_step(index, terminated, truncated)
        info["done"]["success"] = info["success"] = False
        info["done"]["termination_conditions"]["failure"] = {"done": bool(terminated)}
        info["done"]["termination_conditions"]["success"]["done"] = False
        return info

    env.pool._info = with_failure
    observations, rewards, terminated, truncated, infos = env.chunk_step(
        torch.zeros(3, 1, 23)
    )
    assert terminated[:, -1].tolist() == [True, False, False]
    assert truncated[:, -1].tolist() == [False, True, False]
    final = infos[-1]
    assert final["_final_observation"].tolist() == [True, True, False]
    assert not final["final_info"]["episode"]["success_once"].any()
    torch.testing.assert_close(final["final_info"]["episode"]["return"], rewards[:, 0])
    assert final["final_info"]["episode"]["episode_length"].tolist() == [1, 1, 1]
    assert env.elapsed_steps.tolist() == [0, 0, 1]
    assert final["final_observation"]["states"][:, 0].tolist() == [10000, 11000, 12000]
    assert observations[-1]["states"][:, 0].tolist() == [0, 1000, 12000]


@pytest.mark.parametrize("version", [None, "3.9.3", "3.9.3-post2", "3.7.1"])
def test_public_worker_uses_native_simulator_without_starting_sdk(version, monkeypatch):
    from rlinf.envs import get_env_cls

    monkeypatch.setitem(sys.modules, "omnigibson", None)
    preset = (
        Path(__file__).resolve().parents[2]
        / "examples/embodiment/config/env/behavior_r1pro.yaml"
    )
    cfg = OmegaConf.merge(
        OmegaConf.load(preset),
        {"device": "cpu", "enable_offload": True, "enable_init_offload": False},
    )
    if version is not None:
        cfg.simulator_version = version
    worker_env = get_env_cls(cfg.env_type, cfg)
    assert worker_env is get_env_cls("behavior") is behavior_env.BehaviorEnv
    if version == "3.7.1":
        with pytest.raises(ValueError, match="simulator_version"):
            worker_env(cfg, 1, 4, 1, SimpleNamespace(group_world_size=1), False)
    else:
        env = worker_env(cfg, 1, 4, 1, SimpleNamespace(group_world_size=1), False)
        assert env.pool is None and env.seed == cfg.seed + 4
        env.close()


def test_behavior_pool_preserves_shard_and_action_row_order(inline_behavior_actors):
    processes, killed = inline_behavior_actors
    cfg = OmegaConf.create({"num_env_subprocess": 2, "auto_reset": False})
    pool = behavior_env.BehaviorProcessPool(cfg, num_envs=4, seed=40)
    try:
        actions = torch.arange(4 * 2 * 23).reshape(4, 2, 23).float()
        observations, rewards, _, _, _ = pool.env_chunk_step_slice(0, 4, actions)
        torch.testing.assert_close(
            torch.stack([row["robot_r1"]["proprio"][1] for row in observations[-1]]),
            actions[:, -1, 0],
        )
        assert [row["robot_r1"]["proprio"][0] for row in observations[-1]] == [
            40,
            41,
            42,
            43,
        ]
        torch.testing.assert_close(rewards[-1], torch.tensor([0.25, 0.5, 0.25, 0.5]))
        reset_obs, _ = pool.env_reset_slice(0, 4, [3, 1, 2, 0])
        assert [row["robot_r1"]["proprio"][0] for row in reset_obs] == [
            43,
            41,
            42,
            40,
        ]
    finally:
        pool.close()
        pool.close()
    assert len(killed) == 2


def test_behavior_pool_partial_reset_does_not_touch_other_episodes(
    inline_behavior_actors,
):
    processes, _ = inline_behavior_actors
    cfg = OmegaConf.create({"num_env_subprocess": None, "auto_reset": True})
    pool = behavior_env.BehaviorProcessPool(cfg, num_envs=3, seed=40)
    try:
        pool.env_chunk_step_slice(0, 3, torch.zeros(3, 2, 23))
        observations, _ = pool.env_reset_slice(0, 3, [2, 0])
        assert [process.steps.tolist() for process in processes] == [[0], [2], [0]]
        assert [row["robot_r1"]["proprio"][0] for row in observations] == [42, 40]
        assert processes[1].reset_calls == []
    finally:
        pool.close()


def test_behavior_pool_rejects_partial_batch_reset_before_resetting_any_actor(
    inline_behavior_actors,
):
    processes, _ = inline_behavior_actors
    cfg = OmegaConf.create({"num_env_subprocess": 2, "auto_reset": False})
    pool = behavior_env.BehaviorProcessPool(cfg, num_envs=4, seed=40)
    try:
        with pytest.raises(ValueError, match="Partial reset"):
            pool.env_reset_slice(0, 4, [0, 1, 2])
        assert all(process.reset_calls == [] for process in processes)
    finally:
        pool.close()


def test_behavior_pool_rejects_shared_actor_auto_reset(inline_behavior_actors):
    processes, _ = inline_behavior_actors
    cfg = OmegaConf.create({"num_env_subprocess": 1, "auto_reset": True})
    with pytest.raises(ValueError, match="auto_reset"):
        behavior_env.BehaviorProcessPool(cfg, num_envs=2, seed=40)
    assert processes == []


def test_behavior_actor_python_command_runs_with_shell_characters_in_path(
    inline_behavior_actors, monkeypatch, tmp_path
):
    processes, _ = inline_behavior_actors
    python_dir = tmp_path / "simulator & tools"
    python_dir.mkdir()
    python_path = python_dir / "python"
    python_path.symlink_to(sys.executable)
    monkeypatch.setattr(behavior_env.sys, "executable", str(python_path))
    pool = behavior_env.BehaviorProcessPool(
        OmegaConf.create({"auto_reset": False}), num_envs=2, seed=40
    )
    try:
        assert len(processes) == 1
        command = processes[0].actor_options["runtime_env"]["py_executable"]
        assert shlex.split(command) == [str(python_path)]
        result = subprocess.run(
            [
                "/bin/bash",
                "-c",
                command + " -c " + shlex.quote("import sys; print(sys.executable)"),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == str(python_path)
    finally:
        pool.close()


def test_behavior_pool_releases_remaining_actors_if_one_cleanup_fails(
    inline_behavior_actors, monkeypatch
):
    _, killed = inline_behavior_actors
    cfg = OmegaConf.create({"num_env_subprocess": None, "auto_reset": False})
    pool = behavior_env.BehaviorProcessPool(cfg, num_envs=2, seed=40)

    def kill(actor, **kwargs):
        killed.append(actor)
        if len(killed) == 1:
            raise RuntimeError("actor already unavailable")

    monkeypatch.setattr(sys.modules["ray"], "kill", kill)
    with pytest.warns(RuntimeWarning):
        pool.close()
    pool.close()
    assert len(killed) == 2


def test_directory_overrides_preset_split_without_mutating_config(tmp_path):
    (tmp_path / "house_task_turning_on_radio_0_7_template-tro_state.json").write_text(
        "{}"
    )
    cfg = OmegaConf.create(
        {
            "omni_config": {
                "task": {
                    "activity_name": "turning_on_radio",
                    "activity_instance_id": 7,
                    "activity_instance_dir": str(tmp_path),
                    "instance_split": "train",
                }
            }
        }
    )
    loader = ActivityInstanceLoader.from_omni_cfg(cfg.omni_config, num_envs=1, seed=3)
    assert loader.split is None
    assert loader.select([0])[0].instance_id == 7
    assert cfg.omni_config.task.instance_split == "train"


@pytest.fixture
def fake_og_config_dir(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "omnigibson", SimpleNamespace(example_config_path=str(tmp_path))
    )
    monkeypatch.setitem(
        sys.modules, "omnigibson.macros", SimpleNamespace(gm=SimpleNamespace())
    )
    return tmp_path


@pytest.mark.parametrize("explicit_null_proprio", [False, True])
def test_config_preserves_rlinf_rates_controllers_and_legacy_state(
    fake_og_config_dir, explicit_null_proprio
):
    OmegaConf.save(
        OmegaConf.create(
            {
                "env": {},
                "scene": {},
                "task": {},
                "robots": [
                    {
                        "type": "R1Pro",
                        "controller_config": {"base": {"name": "BaseController"}},
                    }
                ],
            }
        ),
        fake_og_config_dir / "r1pro_behavior.yaml",
    )
    cfg = OmegaConf.create(
        {
            "max_episode_steps": 320,
            "omni_config": {
                "env": {
                    "action_frequency": 60,
                    "rendering_frequency": 60,
                    "physics_frequency": 120,
                    "env_wrapper": "rgb",
                },
                "camera": {
                    "head_resolution": [720, 720],
                    "wrist_resolution": [480, 480],
                },
                "robots": [
                    {
                        "name": "robot_r1",
                        "controller_config": {
                            "arm_left": {"use_delta_commands": False}
                        },
                    }
                ],
                "task": {
                    "activity_name": "turning_on_radio",
                    "instance_split": "train",
                    "debug_object_sampling": None,
                },
            },
        }
    )
    if explicit_null_proprio:
        cfg.omni_config.robots[0].proprio_obs = None
    actual = setup_omni_cfg(cfg, 2)
    assert actual["env"]["action_frequency"] == 60
    assert actual["env"]["physics_frequency"] == 120
    assert actual["env"]["num_envs"] == 2
    assert actual["task"]["termination_config"]["max_steps"] == 319
    assert "instance_split" not in actual["task"]
    assert "debug_object_sampling" not in actual["task"]
    robot = actual["robots"][0]
    assert robot["model"] == "r1pro" and "type" not in robot
    assert robot["proprio_obs"] == R1PRO_PROPRIO_KEYS
    assert robot["controller_config"]["base"]["name"] == "BaseController"
    assert robot["controller_config"]["arm_left"]["use_delta_commands"] is False
    assert (
        robot["sensor_config"]["zed_link:Camera:0"]["sensor_kwargs"]["image_height"]
        == 720
    )


def test_legacy_velocity_is_raw_joint_velocity_not_rotated_proprio():
    backend = behavior_env.BehaviorProcess.__new__(behavior_env.BehaviorProcess)
    robot = SimpleNamespace(name="robot_r1", base_control_idx=torch.tensor([0, 1, 5]))
    robot.get_joint_velocities = lambda: torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0, 3.0])
    backend.env = SimpleNamespace(scenes=[SimpleNamespace(robots=[robot])])
    native = torch.arange(256, dtype=torch.float32)
    obs = [{"robot_r1": {"proprio": native}}]
    converted = backend._legacy_observations(obs, [0])[0]["robot_r1"]["proprio"]
    assert torch.equal(converted[253:256], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(converted[:253], native[:253])
    assert torch.equal(native[253:256], torch.tensor([253.0, 254.0, 255.0]))


def test_official_proprio_keeps_robot_frame_base_velocity():
    backend = behavior_env.BehaviorProcess.__new__(behavior_env.BehaviorProcess)
    backend.env = SimpleNamespace(
        scenes=[SimpleNamespace(robots=[SimpleNamespace(name="robot_r1")])]
    )
    native = torch.arange(61, dtype=torch.float32)
    observation = {"robot_r1": {"proprio": native}}
    result = backend._legacy_observations([observation], [0])
    torch.testing.assert_close(result[0]["robot_r1"]["proprio"], native)


@pytest.mark.parametrize(
    "options,skip_frames",
    [
        ({}, False),
        ({"skip_intermediate_obs_in_chunk": True}, True),
        ({"skip_intermediate_renders": True}, True),
        (
            {
                "skip_intermediate_renders": False,
                "skip_intermediate_obs_in_chunk": True,
            },
            False,
        ),
        (
            {
                "skip_intermediate_renders": True,
                "skip_intermediate_obs_in_chunk": False,
            },
            True,
        ),
    ],
)
def test_behavior_process_accepts_both_render_options(
    monkeypatch, options, skip_frames
):
    class Simulator:
        def __init__(self, cfg, num_envs, seed):
            self.position = torch.zeros(num_envs)
            self.image = self.position.clone()

        def step(self, actions, need_obs):
            self.position += actions[:, 0]
            if need_obs:
                self.image = self.position.clone()
            return (
                [{"image": self.image.clone()}] if need_obs else None,
                self.position.clone(),
                torch.zeros(1, dtype=torch.bool),
                torch.zeros(1, dtype=torch.bool),
                [{}],
            )

    process = behavior_env.BehaviorProcess.__new__(behavior_env.BehaviorProcess)
    process.cfg = OmegaConf.create(options)
    monkeypatch.setattr(process, "_step_shard", Simulator(None, 1, 0).step)
    actions = torch.zeros((1, 3, 23))
    actions[0, :, 0] = torch.tensor([1.0, 2.0, 3.0])
    observations, rewards, _, _, _ = process.chunk_step(actions)
    assert torch.equal(torch.stack(rewards), torch.tensor([[1.0], [3.0], [6.0]]))
    assert observations[-1][0]["image"].item() == 6
    if skip_frames:
        assert all(obs is None for obs in observations[:-1])
    else:
        assert [row[0]["image"].item() for row in observations] == [1, 3, 6]


@pytest.mark.parametrize(
    "wrapper,extra_modalities,head_height,wrist_height,task_obs",
    [
        (
            "default",
            {"depth_linear", "seg_semantic", "seg_instance_id"},
            720,
            480,
            False,
        ),
        ("rich_obs", {"normal", "flow"}, 720, 480, True),
        ("rgb_lowres", set(), 224, 224, False),
    ],
)
def test_behavior_config_preserves_existing_observation_wrappers(
    fake_og_config_dir, wrapper, extra_modalities, head_height, wrist_height, task_obs
):
    OmegaConf.save(
        OmegaConf.create(
            {
                "env": {},
                "scene": {},
                "task": {},
                "robots": [{"type": "R1Pro"}],
            }
        ),
        fake_og_config_dir / "r1pro_behavior.yaml",
    )
    cfg = OmegaConf.create(
        {
            "max_episode_steps": 320,
            "omni_config": {
                "env": {"env_wrapper": wrapper},
                "camera": {
                    "head_resolution": [720, 720],
                    "wrist_resolution": [480, 480],
                },
                "robots": [{"name": "robot_r1"}],
                "task": {"activity_name": "turning_on_radio"},
            },
        }
    )
    actual = setup_omni_cfg(cfg, 1)
    robot = actual["robots"][0]
    assert set(robot["obs_modalities"]) == {"rgb", "proprio"} | extra_modalities
    assert robot["proprio_obs"] == R1PRO_PROPRIO_KEYS
    sensors = robot["sensor_config"]
    assert sensors["zed_link:Camera:0"]["sensor_kwargs"]["image_height"] == head_height
    for link in ("left_realsense_link:Camera:0", "right_realsense_link:Camera:0"):
        assert sensors[link]["sensor_kwargs"]["image_height"] == wrist_height
    assert actual["task"]["include_obs"] is task_obs


@pytest.mark.parametrize("with_lights", [False, True])
@pytest.mark.parametrize("ignore_terminations,terminal", [(False, True), (True, False)])
def test_chunk_terminal_frame_is_refreshed_and_episode_stops(
    ignore_terminations, terminal, with_lights
):
    backend = behavior_env.BehaviorProcess.__new__(behavior_env.BehaviorProcess)
    backend.num_envs = 1
    backend._has_reset = True
    backend.ignore_terminations = ignore_terminations
    backend._frozen = torch.zeros(1, dtype=torch.bool)
    backend._terminal = [None]
    backend._last_actions = torch.zeros((1, 23))
    backend._lights = [
        SimpleNamespace(sync_from_current_state=lambda: None) if with_lights else None
    ]
    backend._instance_ids = [7]
    calls = {"steps": 0, "renders": 0}

    def render():
        calls["renders"] += 1

    def step(actions):
        calls["steps"] += 1
        return (
            [{"frame": torch.tensor(0)}],
            torch.ones(1),
            torch.tensor([True]),
            torch.tensor([False]),
            [{}],
        )

    backend.og = SimpleNamespace(
        sim=SimpleNamespace(render=render, render_on_step=lambda enabled: nullcontext())
    )
    backend.env = SimpleNamespace(
        step=step, get_obs=lambda: ([{"frame": torch.tensor(9)}], [{}])
    )
    backend._legacy_observations = lambda observations, indices: observations
    metric = SimpleNamespace(
        timesteps=0, aggregate=lambda **kwargs: {"q_score": {"final": 0.8}}
    )

    def step_metric(**kwargs):
        metric.timesteps += 1

    metric.step = step_metric
    backend.metrics = [metric]
    first = backend._step_shard(torch.zeros((1, 23)), need_obs=False)
    second = backend._step_shard(torch.ones((1, 23)), need_obs=True)
    assert first[0] is None
    if terminal:
        assert calls == {"steps": 1, "renders": 3}
        assert second[0][0]["frame"].item() == 9
        assert first[1].item() == 1 and second[1].item() == 0
        assert second[4][0]["q_score"] == 0.8
        assert first[4][0]["episode_length"] == second[4][0]["episode_length"] == 1
    else:
        assert calls == {"steps": 2, "renders": 3 if with_lights else 0}
        assert not backend._frozen.any()


def test_native_batch_partial_reset_fails_before_simulator_changes():
    backend = behavior_env.BehaviorProcess.__new__(behavior_env.BehaviorProcess)
    backend.num_envs = 2
    with pytest.raises(ValueError, match="share physics"):
        backend.reset([1])


@pytest.mark.parametrize("misplaced_controller", [None, "gripper_left", "arm_right"])
def test_behavior_backend_validates_native_action_and_state_layout(
    monkeypatch, misplaced_controller
):
    monkeypatch.setitem(
        sys.modules,
        "omnigibson.eval.utils.score_utils",
        SimpleNamespace(load_human_stats=lambda activity: {}),
    )
    monkeypatch.setitem(
        sys.modules,
        "omnigibson.metrics.task_metric",
        SimpleNamespace(TaskMetric=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    joints = torch.arange(28, dtype=torch.float32)
    proprio = torch.zeros(256)
    arm_indices = {"left": torch.arange(10, 17), "right": torch.arange(17, 24)}
    gripper_indices = {"left": torch.arange(24, 26), "right": torch.arange(26, 28)}
    trunk_indices = torch.arange(6, 10)
    for region, indices in (
        (slice(158, 165), arm_indices["left"]),
        (slice(193, 195), gripper_indices["left"]),
        (slice(197, 204), arm_indices["right"]),
        (slice(232, 234), gripper_indices["right"]),
        (slice(236, 240), trunk_indices),
    ):
        proprio[region] = joints[indices]
    action_indices = {
        "base": torch.arange(0, 3),
        "trunk": torch.arange(3, 7),
        "arm_left": torch.arange(7, 14),
        "gripper_left": torch.arange(14, 15),
        "arm_right": torch.arange(15, 22),
        "gripper_right": torch.arange(22, 23),
    }
    if misplaced_controller is not None:
        action_indices[misplaced_controller] += 1
    robot = SimpleNamespace(
        model="r1pro",
        action_dim=23,
        n_joints=28,
        controller_action_idx=action_indices,
        get_proprioception=lambda: (proprio, {}),
        get_joint_positions=lambda: joints,
        arm_control_idx=arm_indices,
        gripper_control_idx=gripper_indices,
        trunk_control_idx=trunk_indices,
    )
    monkeypatch.setattr(behavior_env, "apply_runtime_renderer_settings", lambda: None)
    backend = behavior_env.BehaviorProcess.__new__(behavior_env.BehaviorProcess)
    backend.cfg = OmegaConf.create({"omni_config": {"env": {"env_wrapper": None}}})
    backend.env = SimpleNamespace(scenes=[SimpleNamespace(robots=[robot])])
    backend.num_envs = 1
    backend.activity_name = "turning_on_radio"
    if misplaced_controller is not None:
        with pytest.raises(ValueError, match="23-D RLinf action layout"):
            backend._configure_loaded_environment()
    else:
        backend._configure_loaded_environment()
        assert len(backend.metrics) == 1


def test_generated_tro_uses_scene_scope_and_round_trips_robot_pose(
    tmp_path, monkeypatch
):
    from rlinf.envs.sim.behavior.instance_generator import dump_tro_state

    monkeypatch.setitem(
        sys.modules,
        "omnigibson.utils.config_utils",
        SimpleNamespace(TorchEncoder=json.JSONEncoder),
    )
    presampled = {"robot": [{"position": [4, 0, 0], "orientation": [0, 0, 0, 1]}]}
    scene = SimpleNamespace(idx=0, get_task_metadata=lambda key: presampled)
    robot = SimpleNamespace(
        get_position_orientation=lambda frame: ([2, 0, 0], [0, 0, 0, 1])
    )
    obj = SimpleNamespace(dump_state=lambda serialized: {"station": 1})
    task = SimpleNamespace(
        object_scopes=[{"radio.n.01_1": obj, "agent.n.01_1": robot, "missing": None}],
        get_agent=lambda env, env_idx: robot,
    )
    env = SimpleNamespace(scenes=[scene], task=task)
    path = tmp_path / "radio-tro_state.json"
    dump_tro_state(env, path, overwrite=False)
    assert json.loads(path.read_text()) == {
        "radio.n.01_1": {"station": 1},
        "robot_poses": presampled,
    }
    with pytest.raises(FileExistsError):
        dump_tro_state(env, path, overwrite=False)
    dump_tro_state(env, path, overwrite=True, capture_current_robot_pose=True)
    assert json.loads(path.read_text())["robot_poses"]["robot"][0]["position"] == [
        2,
        0,
        0,
    ]
    assert presampled["robot"][0]["position"] == [4, 0, 0]


def test_generator_validates_bddl_predicates_in_its_scene(monkeypatch):
    from rlinf.envs.sim.behavior.instance_generator import _validate_sample

    macros = SimpleNamespace(ENABLE_TRANSITION_RULES=False, unlocked=nullcontext)
    monkeypatch.setitem(sys.modules, "omnigibson.macros", SimpleNamespace(gm=macros))
    calls = []

    def evaluate_predicate(scene_index, predicate, *entities):
        calls.append((scene_index, predicate, entities))
        return True

    task = SimpleNamespace(
        _base_conditions=object(),
        activity_initial_conditions=object(),
        _evaluate_predicate=evaluate_predicate,
    )
    env = SimpleNamespace(
        task=task, scenes=[SimpleNamespace(load_room_instances=["bedroom_0"])]
    )

    def validate(view, sampled, stable, active_room_instances):
        assert sampled == {"state": {"value": 1}} and stable == {"value": 0}
        assert active_room_instances == ["bedroom_0"]
        assert view.activity_initial_conditions is task.activity_initial_conditions
        assert view._evaluate_predicate("ontop", "radio", "table")
        macros.ENABLE_TRANSITION_RULES = True
        raise ValueError("unstable scene")

    with pytest.raises(ValueError, match="unstable scene"):
        _validate_sample(env, {"value": 1}, {"value": 0}, validate)
    assert calls == [(0, "ontop", ("radio", "table"))]
    assert task._evaluate_predicate is evaluate_predicate
    assert not macros.ENABLE_TRANSITION_RULES
