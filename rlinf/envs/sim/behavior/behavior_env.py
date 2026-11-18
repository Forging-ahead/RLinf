# Copyright 2025 The RLinf Authors.
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

"""RLinf worker interface for the native BEHAVIOR 3.9.3 simulator."""

from __future__ import annotations

import gc
import json
import os
import random
import shlex
import sys
import time
import warnings
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from threading import get_ident
from typing import TYPE_CHECKING, Any, ClassVar

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .instance_loader import ActivityInstanceLoader
from .utils import (
    CAMERA_LINKS,
    configure_macros,
    convert_uint8_rgb,
    make_environment_config,
    set_head_camera_aperture,
    validate_simulator_version,
)

if TYPE_CHECKING:
    from omnigibson.envs.env_base import Environment
    from omnigibson.robots.robot import Robot


def _cpu(value: Any) -> Any:
    """Materialize simulator tensors before crossing the process boundary."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(item) for item in value)
    return value


class BehaviorProcess:
    """Own a native simulator shard in one Ray actor."""

    def __init__(self, cfg: Any, num_envs: int, seed: int):
        self.backend = BehaviorSimulator(cfg, num_envs, seed)
        # Isaac Sim's `omni.kit.app` calls ``gc.disable()`` at startup.
        # OmniGibson has self-referential cycles and leaks memory when
        # cyclic GC is disabled. Since we do not need real-time performance,
        # enable cyclic GC here so that we do not encounter OOMs in long runs.
        gc.enable()

        # Upstream renamed this option; an explicit modern value wins, even False.
        self.skip_obs = bool(
            cfg.get(
                "skip_intermediate_renders",
                cfg.get("skip_intermediate_obs_in_chunk", False),
            )
        )

    def get_activity_name(self) -> str:
        return self.backend.activity_name

    def reset(self, env_indices: list[int] | None = None) -> tuple[list, list]:
        return _cpu(self.backend.reset(env_indices))

    def _step_shard(self, actions: torch.Tensor, need_obs: bool):
        observations, rewards, terms, truncs, infos = self.backend.step(
            actions, render=need_obs, get_obs=need_obs
        )
        # Intermediate frames do not cross the simulator process boundary.
        return _cpu((observations, rewards, terms, truncs, infos))

    def chunk_step(self, actions: torch.Tensor) -> tuple[list, ...]:
        columns: tuple[list, ...] = ([], [], [], [], [])
        for t in range(actions.shape[1]):
            need_obs = not self.skip_obs or t == actions.shape[1] - 1
            result = self._step_shard(actions[:, t], need_obs)
            for column, value in zip(columns, result):
                column.append(value)
        return columns

    def close(self) -> None:
        self.backend.close()


class BehaviorProcessPool:
    """Share one pool among the pipeline stages of a worker's train or eval config."""

    _shared_pools: ClassVar[dict[tuple[int, int], BehaviorProcessPool]] = {}

    @classmethod
    def acquire_shared(
        cls, cfg, worker_info, pipeline_stage_num, num_envs, seed_offset
    ):
        # EnvWorker assigns rank * stages + stage_id. A reopened stage keeps its slice.
        rank, stage = divmod(seed_offset, pipeline_stage_num)
        key = (id(cfg), rank)
        pool = cls._shared_pools.get(key)
        if pool is None:
            worker_envs = int(cfg.total_num_envs) // worker_info.group_world_size
            if worker_envs != num_envs * pipeline_stage_num:
                raise ValueError(
                    "BEHAVIOR environments must divide evenly among worker stages."
                )
            seed = int(cfg.seed) + rank * worker_envs
            pool = cls(cfg, worker_envs, seed, pipeline_stage_num)
            pool._key = key
            cls._shared_pools[key] = pool
        offset = stage * num_envs
        if offset in pool._active_offsets:
            raise RuntimeError(f"BEHAVIOR stage at offset {offset} is already in use.")
        pool._active_offsets.add(offset)
        return pool, offset

    def release_shared(self, offset: int) -> None:
        self._active_offsets.remove(offset)
        if not self._active_offsets:
            del self._shared_pools[self._key]
            self.close()

    def __init__(self, cfg: Any, num_envs: int, seed: int, pipeline_stage_num: int = 1):
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        self.ray = ray
        self.cfg = cfg
        self._active_offsets: set[int] = set()
        requested = cfg.get("num_env_subprocess", 1)
        count = num_envs if requested is None else int(requested)
        if not 1 <= count <= num_envs:
            raise ValueError("num_env_subprocess must be null or in [1, num_envs].")
        if cfg.auto_reset and count != num_envs:
            raise ValueError(
                "BEHAVIOR 3.9.3 auto_reset requires one actor per environment. "
                "Set num_env_subprocess: null to select this automatically."
            )
        if count < pipeline_stage_num:
            raise ValueError(
                "BEHAVIOR 3.9.3 advances all scenes in a simulator. "
                f"{pipeline_stage_num} independent pipeline stages need at least "
                f"{pipeline_stage_num} simulation subprocesses per worker; "
                f"num_env_subprocess is {count}."
            )
        self.num_envs = num_envs
        self.skip_intermediate_renders = bool(
            cfg.get(
                "skip_intermediate_renders",
                cfg.get("skip_intermediate_obs_in_chunk", False),
            )
        )
        # Keep the configured process budget; an actor's native physics clock may
        # advance only one stage. Distribute leftover actors/rows within those bounds.
        envs_per_stage = num_envs // pipeline_stage_num
        base_count, extra = divmod(count, pipeline_stage_num)
        self.rows = []
        for stage in range(pipeline_stage_num):
            stage_count = base_count + (stage < extra)
            base_rows, extra_rows = divmod(envs_per_stage, stage_count)
            start = stage * envs_per_stage
            for shard in range(stage_count):
                stop = start + base_rows + (shard < extra_rows)
                self.rows.append(range(start, stop))
                start = stop
        self._actor_for_row = [sp for sp, rows in enumerate(self.rows) for _ in rows]
        self.env_processes: list = []
        # Ray children must retain the environment worker's assigned GPU. They
        # do not reserve new GPUs independently of RLinf's placement system.
        env_vars = {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            env_vars["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
        actor = ray.remote(num_cpus=1)(BehaviorProcess)
        node_id = ray.get_runtime_context().get_node_id()
        actor = actor.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id, soft=False
            ),
            runtime_env={
                "py_executable": shlex.quote(sys.executable),
                "env_vars": env_vars,
            },
        )
        startup = cfg.get("behavior", {})
        attempts = max(1, int(startup.get("init_retry_count", 3)))
        delay = float(startup.get("init_retry_delay", 5.0))
        for attempt in range(1, attempts + 1):
            try:
                for rows in self.rows:
                    self.env_processes.append(
                        actor.remote(cfg, len(rows), seed + rows.start)
                    )
                names = ray.get(
                    [proc.get_activity_name.remote() for proc in self.env_processes]
                )
                if len(set(names)) != 1:
                    raise RuntimeError(
                        f"BEHAVIOR actors loaded different tasks: {names}"
                    )
                self.activity_name = names[0]
                break
            except BaseException as error:
                self.close()
                if attempt == attempts or not isinstance(error, Exception):
                    raise
                warnings.warn(
                    f"BEHAVIOR actor startup failed ({error}); retrying in {delay}s.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                time.sleep(delay)
                delay *= float(startup.get("init_retry_backoff", 2.0))

    def _slice_plan(self, global_start, num_envs, env_indices=None):
        """Map requested slice positions to local rows, preserving caller order."""
        selected = list(range(num_envs) if env_indices is None else env_indices)
        if len(set(selected)) != len(selected) or any(
            i < 0 or i >= num_envs for i in selected
        ):
            raise ValueError("reset indices must be unique valid environment indices.")
        by_process: dict[int, tuple[list[int], list[int]]] = {}
        for pos, index in enumerate(selected):
            row = global_start + index
            sp = self._actor_for_row[row]
            positions, local_rows = by_process.setdefault(sp, ([], []))
            positions.append(pos)
            local_rows.append(row - self.rows[sp].start)
        plan = []
        for sp, (positions, local_rows) in by_process.items():
            if len(local_rows) != len(self.rows[sp]):
                raise ValueError(
                    "Partial reset within a native 3.9.3 batch advances other "
                    "scenes. Use num_env_subprocess: null for independent reset."
                )
            plan.append((sp, positions, local_rows))
        return plan

    def env_reset_slice(self, global_start, num_envs, env_indices=None):
        plan = self._slice_plan(global_start, num_envs, env_indices)
        refs = [self.env_processes[sp].reset.remote(local) for sp, _, local in plan]
        count = num_envs if env_indices is None else len(env_indices)
        raw_obs, infos = [None] * count, [None] * count
        for (_, positions, _), (obs, info) in zip(plan, self.ray.get(refs)):
            for local, pos in enumerate(positions):
                raw_obs[pos], infos[pos] = obs[local], info[local]
        return raw_obs, infos

    def env_chunk_step_slice(self, global_start, num_envs, chunk_actions):
        plan = self._slice_plan(global_start, num_envs)
        refs = [
            self.env_processes[sp].chunk_step.remote(chunk_actions[positions])
            for sp, positions, _ in plan
        ]
        return self._merge_shards(
            self.ray.get(refs), plan, num_envs, chunk_actions.shape[1]
        )

    def _merge_shards(
        self,
        shard_results: list,
        plan: list[tuple[int, list[int], list[int]]],
        slice_num_envs: int,
        chunk_size: int,
    ):
        """Gather per-subprocess shard outputs into ``[chunk][slice]`` order."""
        merged_obs: list = []
        merged_rewards: list = []
        merged_terms: list = []
        merged_trunc: list = []
        merged_infos: list = []
        for t in range(chunk_size):
            is_last = t == chunk_size - 1
            need_obs = not self.skip_intermediate_renders or is_last
            obs_t: list | None = [None] * slice_num_envs if need_obs else None
            reward_t = torch.zeros(slice_num_envs, dtype=torch.float32)
            term_t = torch.zeros(slice_num_envs, dtype=torch.bool)
            trunc_t = torch.zeros(slice_num_envs, dtype=torch.bool)
            info_t: list = [{} for _ in range(slice_num_envs)]
            for (obs_per_t, rewards_per_t, terms_per_t, truncs_per_t, infos_per_t), (
                _sp,
                positions,
                _local_rows,
            ) in zip(shard_results, plan):
                obs_at_t = obs_per_t[t]
                rewards_at_t = rewards_per_t[t]
                terms_at_t = terms_per_t[t]
                truncs_at_t = truncs_per_t[t]
                infos_at_t = infos_per_t[t]
                for i, pos in enumerate(positions):
                    if need_obs:
                        obs_t[pos] = obs_at_t[i]
                    reward_t[pos] = float(rewards_at_t[i])
                    term_t[pos] = bool(terms_at_t[i])
                    trunc_t[pos] = bool(truncs_at_t[i])
                    info_t[pos] = infos_at_t[i]
            merged_obs.append(obs_t)
            merged_rewards.append(reward_t)
            merged_terms.append(term_t)
            merged_trunc.append(trunc_t)
            merged_infos.append(info_t)
        return merged_obs, merged_rewards, merged_terms, merged_trunc, merged_infos

    def close(self) -> None:
        """Release only actors owned by this pool, including failed startups."""
        processes, self.env_processes = self.env_processes, []
        # Killing a dedicated simulator actor also releases Kit/CUDA resources;
        # waiting for a failed actor's graceful shutdown can block indefinitely.
        errors = []
        for process in processes:
            try:
                self.ray.kill(process, no_restart=True)
            except Exception as error:
                errors.append(str(error))
        if errors:
            warnings.warn(
                f"Failed to release some BEHAVIOR simulator actors: {errors}",
                RuntimeWarning,
                stacklevel=2,
            )


class BehaviorEnv(gym.Env):
    """Batched BEHAVIOR environment for RLinf workers.

    Args:
        cfg: RLinf BEHAVIOR configuration for OmniGibson 3.9.3.
        num_envs: Number of environments owned by this pipeline stage.
        seed_offset: Worker/stage seed offset.
        total_num_processes: Environment worker count including pipeline stages.
        worker_info: RLinf worker identity and group size.
        record_metrics: Whether to include per-episode tensors in returned info.
    """

    def __init__(
        self,
        cfg: Any,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
        record_metrics: bool = True,
    ) -> None:
        validate_simulator_version(cfg)
        self.cfg = cfg
        self.reward_coef = cfg.get("reward_coef", 1)

        self.num_envs = num_envs
        self.ignore_terminations = cfg.ignore_terminations
        self.seed_offset = seed_offset
        self.seed = int(self.cfg.seed) + seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.record_metrics = record_metrics
        self._is_start = True
        self.enable_offload = cfg.get("enable_offload", False)
        self.enable_init_offload = cfg.get("enable_init_offload", True)
        self.pool = None
        self.pool_offset = None
        self.task_description = None
        if num_envs < 1:
            raise ValueError("num_envs must be positive.")
        if total_num_processes % worker_info.group_world_size != 0:
            raise ValueError(
                f"total_num_processes ({total_num_processes}) must be divisible by "
                f"worker_info.group_world_size ({worker_info.group_world_size}) to infer pipeline_stage_num."
            )
        self.pipeline_stage_num = total_num_processes // worker_info.group_world_size

        self.auto_reset = cfg.auto_reset
        self.max_episode_steps = torch.tensor(cfg.max_episode_steps)
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        if self.auto_reset and self.enable_offload:
            raise ValueError(
                "BEHAVIOR 3.9.3 cannot combine auto_reset with enable_offload: "
                "offload releases simulator state, while automatic reset keeps "
                "the worker's previous observation. Disable one of these options."
            )
        self._device = cfg.get("device", "cuda")
        self._requires_reset = True
        self._steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        if self.record_metrics:
            self._init_metrics()
        if not (self.enable_offload and not self.enable_init_offload):
            self._init_env()

    def _ensure_pool(self) -> None:
        if self.pool is None:
            self.pool, self.pool_offset = BehaviorProcessPool.acquire_shared(
                self.cfg,
                self.worker_info,
                self.pipeline_stage_num,
                self.num_envs,
                self.seed_offset,
            )

    def _init_env(self) -> None:
        self._ensure_pool()
        try:
            self._load_tasks_cfg(self.pool.activity_name)
        except BaseException:
            self.close()
            raise

    def _load_tasks_cfg(self, activity_name: str):
        # Read task description

        task_description_path = os.path.join(
            os.path.dirname(__file__), "behavior_task.jsonl"
        )
        with open(task_description_path, "r", encoding="utf-8") as f:
            text = f.read()
            task_description = [
                json.loads(x) for x in text.strip().split("\n") if x.strip()
            ]
        task_description_map = {
            task_description[i]["task_name"]: task_description[i]["task"]
            for i in range(len(task_description))
        }
        self.task_description = task_description_map[activity_name]

    def env_reset(self, env_indices: list[int] | None = None) -> tuple[list, list]:
        """Return raw simulator observations and per-environment info."""
        if self.pool is None:
            self._init_env()
        # A failed fan-out may already have reset some actors. Only a successful
        # full-stage reset can make such a stage safe to step again.
        already_needed_reset = self._requires_reset
        self._requires_reset = True
        result = self.pool.env_reset_slice(self.pool_offset, self.num_envs, env_indices)
        full_reset = env_indices is None or len(env_indices) == self.num_envs
        self._requires_reset = already_needed_reset and not full_reset
        return result

    def env_chunk_step(self, chunk_actions: torch.Tensor) -> tuple[list, ...]:
        """Return raw per-step simulator results for one action chunk."""
        if self._requires_reset:
            raise RuntimeError(
                "Call reset() for the entire BEHAVIOR stage before stepping."
            )
        # ray.get can fail after another shard accepted its action. Retrying
        # the chunk in that state would execute that shard's action twice.
        self._requires_reset = True
        result = self.pool.env_chunk_step_slice(
            self.pool_offset, self.num_envs, chunk_actions
        )
        self._requires_reset = False
        return result

    def _extract_obs_image(self, raw_obs: dict) -> dict:
        sensors = next(
            (
                value
                for value in raw_obs.values()
                if isinstance(value, dict) and "proprio" in value
            ),
            None,
        )
        if sensors is None:
            raise ValueError("BEHAVIOR observation requires robot proprioception.")
        state = torch.as_tensor(sensors["proprio"])
        if state.shape not in ((61,), (256,)):
            raise ValueError(
                f"Expected 61 or 256 R1Pro state values, got {tuple(state.shape)}."
            )
        images = []
        for role in ("head", "left_wrist", "right_wrist"):
            camera = CAMERA_LINKS[role]
            matches = [value for key, value in sensors.items() if camera in key]
            if len(matches) != 1 or "rgb" not in matches[0]:
                raise ValueError(f"Expected one RGB sensor matching {camera}.")
            images.append(convert_uint8_rgb(matches[0]["rgb"]))
        return {
            "main_images": images[0],
            "wrist_images": torch.stack(images[1:]),
            "state": state,
        }

    def _wrap_obs(self, obs_list):
        extracted_obs_list = []
        for obs in obs_list:
            extracted_obs = self._extract_obs_image(obs)
            extracted_obs_list.append(extracted_obs)

        obs = {
            "main_images": torch.stack(
                [obs["main_images"] for obs in extracted_obs_list], axis=0
            ),  # [N_ENV, H, W, C]
            "wrist_images": torch.stack(
                [obs["wrist_images"] for obs in extracted_obs_list], axis=0
            ),  # [N_ENV, N_IMG, H, W, C]
            "task_descriptions": [self.task_description] * len(extracted_obs_list),
            "states": torch.stack(
                [obs["state"] for obs in extracted_obs_list], axis=0
            ),  # [N_ENV, state_dim]
        }
        return obs

    def _calc_step_reward(self, reward):
        return self.reward_coef * reward

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, indices: list[int]) -> None:
        self._steps[indices] = 0
        if not self.record_metrics:
            return
        self.success_once[indices] = False
        self.returns[indices] = 0

    def _record_metrics(self, rewards: torch.Tensor, infos: list[dict]) -> dict:
        if not self.record_metrics:
            return {}
        successes = torch.tensor(
            [
                bool(info.get("done", {}).get("success", info.get("success", False)))
                for info in infos
            ],
            dtype=torch.bool,
            device=self.device,
        )
        self.success_once |= successes
        self.returns += rewards.to(self.device)
        lengths = torch.tensor(
            [
                int(info.get("episode_length", self._steps[index].item()))
                for index, info in enumerate(infos)
            ],
            dtype=torch.long,
            device=self.device,
        )
        episode = {
            "success": successes,
            "episode_length": lengths,
            "success_once": self.success_once.clone(),
            "return": self.returns.clone(),
            "episode_len": lengths.clone(),
            "reward": self.returns / lengths.clamp_min(1),
        }
        if self.ignore_terminations:
            episode["success_at_end"] = successes.clone()
        # TaskMetric has no score immediately after reset. Publish it only
        # after every row has a real score from a completed simulator step.
        if all(info.get("q_score") is not None for info in infos):
            episode["q_score"] = torch.tensor(
                [float(info["q_score"]) for info in infos],
                device=self.device,
            )
        return {"episode": episode}

    def reset(self) -> tuple[dict, dict]:
        """Reset all owned episodes and return the original RLinf observation keys."""
        raw, infos = self.env_reset()
        self._reset_metrics(list(range(self.num_envs)))
        return self._wrap_obs(raw), self._record_metrics(
            torch.zeros(self.num_envs), infos
        )

    def chunk_step(
        self, chunk_actions: Any
    ) -> tuple[list, torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """Execute native 23-D actions [N,T,23]; report done at the chunk boundary."""
        actions = torch.as_tensor(chunk_actions).detach().cpu()
        if (
            actions.ndim != 3
            or actions.shape[0] != self.num_envs
            or actions.shape[1] < 1
            or actions.shape[2] != 23
        ):
            raise ValueError(
                f"Expected actions [{self.num_envs}, T>=1, 23], got {tuple(actions.shape)}."
            )
        if not torch.isfinite(actions).all():
            raise ValueError("BEHAVIOR actions must be finite.")
        raw_obs, rewards, terms, truncs, raw_infos = self.env_chunk_step(actions)
        obs_list, infos_list = [], []
        reward_tensor = self._calc_step_reward(
            torch.stack(
                [torch.as_tensor(value, dtype=torch.float32) for value in rewards],
                dim=1,
            )
        )
        term_tensor = torch.stack(
            [torch.as_tensor(value, dtype=torch.bool) for value in terms], dim=1
        )
        trunc_tensor = torch.stack(
            [torch.as_tensor(value, dtype=torch.bool) for value in truncs], dim=1
        )
        if self.ignore_terminations:
            term_tensor.zero_()
        for index, (obs, infos) in enumerate(zip(raw_obs, raw_infos)):
            self._steps[:] = torch.tensor(
                [
                    int(info.get("episode_length", self._steps[row].item() + 1))
                    for row, info in enumerate(infos)
                ],
                device=self.device,
            )
            obs_list.append(None if obs is None else self._wrap_obs(obs))
            infos_list.append(self._record_metrics(reward_tensor[:, index], infos))
        if not obs_list or obs_list[-1] is None:
            raise RuntimeError(
                "BEHAVIOR must return the final observation of every chunk."
            )
        past_terms, past_truncs = term_tensor.any(dim=1), trunc_tensor.any(dim=1)
        done = past_terms | past_truncs
        if self.auto_reset and done.any():
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                done, obs_list[-1], infos_list[-1]
            )
        term_tensor.zero_()
        trunc_tensor.zero_()
        term_tensor[:, -1] = past_terms
        trunc_tensor[:, -1] = past_truncs
        return obs_list, reward_tensor, term_tensor, trunc_tensor, infos_list

    def _handle_auto_reset(
        self, dones: torch.Tensor, final_obs: dict, final_info: dict
    ) -> tuple[dict, dict]:
        indices = dones.nonzero(as_tuple=False).flatten().tolist()
        raw, _infos = self.env_reset(indices)
        reset_obs = self._wrap_obs(raw)
        current_obs = {}
        for key, value in final_obs.items():
            merged = value.clone() if isinstance(value, torch.Tensor) else list(value)
            for local, index in enumerate(indices):
                merged[index] = reset_obs[key][local]
            current_obs[key] = merged
        self._reset_metrics(indices)
        current_info = (
            {
                "episode": {
                    key: value.clone() for key, value in final_info["episode"].items()
                }
            }
            if self.record_metrics
            else {}
        )
        if self.record_metrics:
            for value in current_info["episode"].values():
                value[indices] = 0
        current_info.update(
            final_observation=final_obs,
            final_info=final_info,
            _final_observation=dones.clone(),
            _final_info=dones.clone(),
            _elapsed_steps=dones.clone(),
        )
        return current_obs, current_info

    @property
    def device(self) -> str:
        return self._device

    @property
    def elapsed_steps(self) -> torch.Tensor:
        return self._steps.clone()

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = value

    def update_reset_state_ids(self) -> None:
        """Retain the existing single-task reset-state hook."""

    def offload(self) -> None:
        """Release simulator resources; the next reset recreates them."""
        self.close()

    def close(self) -> None:
        """Close owned simulator processes; repeated calls are harmless."""
        self._requires_reset = True
        if self.pool is not None:
            pool, offset = self.pool, self.pool_offset
            self.pool = self.pool_offset = None
            pool.release_shared(offset)


def snapshot(value: Any) -> Any:
    """Own simulator-backed tensors before the next physics step mutates them."""
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(snapshot(item) for item in value)
    return deepcopy(value)


class BehaviorSimulator:
    """One process owns one native Environment and its episode accounting.

    Use one logical scene per process for independently resetting RLinf envs.
    A native multi-scene batch is supported for synchronized episode boundaries;
    partial reset is rejected because reset/settling advances shared physics.
    All calls and borrowed native handles belong to the constructing thread.
    Close is the final operation in this dedicated process: official SDK
    shutdown may terminate Python instead of returning. Save results and errors
    and flush logs first; the process owner verifies exit and resource release.
    Create the next instance in a fresh process.
    If construction fails, the original error is propagated after a best-effort
    simulation stop. The process owner must retire that process to release SDK
    resources; partial initialization is not safe for application teardown.

    Args:
        cfg: RLinf BEHAVIOR configuration with native ``omni_config`` overrides.
        num_envs: Number of scenes sharing one simulator clock.
        seed: Initial random seed and instance-selection seed.
    """

    def __init__(self, cfg: DictConfig, num_envs: int = 1, seed: int = 0) -> None:
        validate_simulator_version(cfg)
        self._owner_thread = get_ident()
        self._closed = False
        self._owns_simulator = False
        self.env: Environment | None = None
        self.cfg = cfg
        self.num_envs = int(num_envs)
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive.")
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        threads = OmegaConf.select(cfg, "torch_num_threads")
        if threads is not None:
            if int(threads) < 1:
                raise ValueError("torch_num_threads must be positive.")
            torch.set_num_threads(int(threads))

        # Imports remain in the simulator actor: importing OG in the parent
        # worker would launch/register simulator resources in the wrong process.
        import omnigibson as og

        if str(getattr(og, "__version__", "")).split("-")[0] != "3.9.3":
            raise RuntimeError(
                "BehaviorSimulator requires OmniGibson 3.9.3 (v3.9.3-post2 API); "
                f"found {getattr(og, '__version__', 'unknown')!r}."
            )
        self.og = og
        if og.sim is not None or getattr(og, "app", None) is not None:
            raise RuntimeError(
                "BehaviorSimulator requires its own process; an OmniGibson "
                "simulator/application already exists here."
            )
        configure_macros(cfg)
        self.instances = ActivityInstanceLoader(cfg, self.num_envs, seed)
        self.activity_name = self.instances.activity_name
        self.config = make_environment_config(
            cfg, self.num_envs, str(Path(og.__file__).parent)
        )
        self.config = self.instances.initial_config(self.config)
        self.ignore_terminations = bool(
            OmegaConf.select(cfg, "ignore_terminations", default=False)
        )
        self._has_reset = False
        self._reset_required = True
        self._execution_outcome_known = True
        self._frozen = torch.zeros(self.num_envs, dtype=torch.bool)
        self._terminal = [None] * self.num_envs
        self._last_actions = torch.zeros((self.num_envs, 23), dtype=torch.float32)
        self._last_executed_actions: list[torch.Tensor | None] = [None] * self.num_envs
        self._metrics_snapshot: list[dict[str, Any]] = [
            {} for _ in range(self.num_envs)
        ]
        self._instance_ids = [self.instances.instance_id] * self.num_envs
        self._owns_simulator = True
        try:
            self.env = og.Environment(configs=self.config)
            og.sim.update_handles()
            self.instances.bind(self.env)
            self._configure_loaded_environment()
        except BaseException:
            # Kit teardown may exit or crash before Python can report the
            # construction error. Stop only; the process owner reclaims SDK
            # resources after it receives the original error.
            self._closed = True
            self._owns_simulator = False
            try:
                if og.sim is not None:
                    og.sim.stop()
            except BaseException:
                pass
            raise

    def _check_access(self, *, require_reset: bool = False) -> None:
        if self._owner_thread != get_ident():
            raise RuntimeError("Use BehaviorSimulator from its constructing thread.")
        if self._closed:
            raise RuntimeError("BehaviorSimulator is closed.")
        if require_reset and not self._has_reset:
            raise RuntimeError("Call reset() before reading episode observations.")

    @property
    def native_env(self) -> Environment:
        """Borrow the native environment for same-thread SDK operations.

        The handle expires on reset that rebuilds the scene, or on close.
        Stepping/resetting this handle bypasses this adapter's metrics and
        command accounting; advance episodes through this adapter instead.
        """
        self._check_access()
        return self.env

    def get_robot(self, env_index: int = 0) -> Robot:
        """Borrow one native R1Pro, including its public controllers and sensors.

        Args:
            env_index: Scene index in this native batch.

        Returns:
            The live robot, valid until a rebuilding reset or close.
        """
        self._check_access()
        if type(env_index) is not int or not 0 <= env_index < self.num_envs:
            raise ValueError(f"env_index must be an integer in [0, {self.num_envs}).")
        return self.env.scenes[env_index].robots[0]

    def get_robot_state(self, env_index: int = 0) -> dict[str, Any]:
        """Copy native joints, velocities, poses and last successful step input.

        Joint arrays use the physical 28-joint order in ``joint_names``.
        Poses are metres and xyzw; revolute/prismatic joints use radian/metre.
        Base velocities contain m/s, m/s, rad/s. Raw virtual-joint velocities
        match legacy256; robot-frame velocities additionally apply base yaw.
        ``last_executed_action`` is a native23 input, not a controller target
        or measured state. After a failed step it is historical; consult
        ``execution_outcome_known`` and reset before further stepping.

        Args:
            env_index: Scene index in this native batch.

        Returns:
            Owned tensors and dictionaries from the live simulator.
        """
        self._check_access(require_reset=True)
        robot = self.get_robot(env_index)
        positions = robot.get_joint_positions()
        velocities = robot.get_joint_velocities()
        base_velocity = velocities[robot.base_control_idx]
        yaw = positions[robot.base_idx][5]
        c, s = torch.cos(yaw), torch.sin(yaw)
        local_velocity = torch.stack(
            (
                c * base_velocity[0] + s * base_velocity[1],
                -s * base_velocity[0] + c * base_velocity[1],
                base_velocity[2],
            )
        )

        def pose(values: tuple) -> dict[str, Any]:
            return {"position": values[0], "orientation_xyzw": values[1]}

        return snapshot(
            {
                "robot_name": robot.name,
                "joint_names": list(robot.joints),
                "joint_positions": positions,
                "joint_velocities": velocities,
                "world_pose": pose(robot.get_position_orientation()),
                "eef_poses": {
                    arm: {
                        "world": pose(robot.get_eef_pose(arm=arm)),
                        "robot_base": pose(robot.get_relative_eef_pose(arm=arm)),
                    }
                    for arm in ("left", "right")
                },
                "raw_virtual_joint_velocity": base_velocity,
                "robot_frame_base_velocity": local_velocity,
                "last_executed_action": self._last_executed_actions[env_index],
                "execution_outcome_known": self._execution_outcome_known,
                "units": {
                    "position": "metre",
                    "joint_position": "radian (revolute), metre (prismatic)",
                    "joint_velocity": "radian/s (revolute), metre/s (prismatic)",
                    "base_velocity": ["metre/s", "metre/s", "radian/s"],
                },
                "quaternion_order": "xyzw",
                "state_source": "live_simulator",
            }
        )

    def get_camera_metadata(self, env_index: int = 0) -> dict[str, dict[str, Any]]:
        """Copy sensor calibration and current world poses without physics steps.

        OpenGL camera axes are +X right, +Y up and -Z forward. The returned pose
        is current; read it immediately after a rendered reset or step when
        pairing it with an image. The SDK may
        initialize camera annotators by rendering on the first calibration read.

        Args:
            env_index: Scene index in this native batch.

        Returns:
            Metadata keyed by head, left_wrist and right_wrist.
        """
        self._check_access(require_reset=True)
        robot = self.get_robot(env_index)
        result = {}
        for role, link in CAMERA_LINKS.items():
            name = f"{robot.name}:{link}"
            sensor = robot.sensors[name]
            position, orientation = sensor.get_position_orientation()
            result[role] = {
                "sensor_name": name,
                "intrinsic_matrix": sensor.intrinsic_matrix,
                "height": int(sensor.image_height),
                "width": int(sensor.image_width),
                "world_pose": {"position": position, "orientation_xyzw": orientation},
                "camera_frame": "opengl",
                "camera_axes": {"x": "right", "y": "up", "forward": "-z"},
                "pose_frame": "world",
                "length_unit": "metre",
                "quaternion_order": "xyzw",
            }
        return snapshot(result)

    def get_metrics(self) -> list[dict[str, Any]]:
        """Copy metrics captured at the last completed reset/step, without updating them."""
        self._check_access(require_reset=True)
        return snapshot(self._metrics_snapshot)

    def _configure_loaded_environment(self) -> None:
        from omnigibson.eval.utils.score_utils import load_human_stats
        from omnigibson.metrics.task_metric import TaskMetric

        for scene in self.env.scenes:
            if len(scene.robots) != 1:
                raise ValueError("RLinf BEHAVIOR expects exactly one robot per scene.")
            robot = scene.robots[0]
            if robot.model != "r1pro" or robot.action_dim != 23 or robot.n_joints != 28:
                raise ValueError(
                    "R1Pro must expose the legacy 28 joints and 23 native action dimensions."
                )
            expected_actions = {
                "base": (0, 3),
                "trunk": (3, 7),
                "arm_left": (7, 14),
                "gripper_left": (14, 15),
                "arm_right": (15, 22),
                "gripper_right": (22, 23),
            }
            action_indices = robot.controller_action_idx
            for controller, (start, stop) in expected_actions.items():
                if controller not in action_indices or not torch.equal(
                    action_indices[controller].cpu(), torch.arange(start, stop)
                ):
                    raise ValueError(
                        f"Controller {controller!r} does not match the native "
                        f"23-D RLinf action layout at [{start}:{stop}]."
                    )
            proprio, _ = robot.get_proprioception()
            if proprio.shape == (256,):
                # Validate the legacy positions against the live robot. The SDK
                # defines the field order of its compact 61-value observation.
                joints = robot.get_joint_positions()
                checks = (
                    (slice(158, 165), robot.arm_control_idx["left"]),
                    (slice(193, 195), robot.gripper_control_idx["left"]),
                    (slice(197, 204), robot.arm_control_idx["right"]),
                    (slice(232, 234), robot.gripper_control_idx["right"]),
                    (slice(236, 240), robot.trunk_control_idx),
                )
                for region, indices in checks:
                    if not torch.allclose(proprio[region], joints[indices]):
                        raise ValueError(
                            f"R1Pro proprio field {region} no longer matches the RLinf model contract."
                        )
            elif proprio.shape != (61,):
                raise ValueError(
                    f"Unexpected R1Pro state shape {tuple(proprio.shape)}."
                )
            wrapper = OmegaConf.select(
                self.cfg, "omni_config.env.env_wrapper", default="rgb"
            )
            if wrapper is not None:
                set_head_camera_aperture(robot)
        human_stats = load_human_stats(self.activity_name)
        self.metrics = [
            TaskMetric(human_stats=human_stats, env_idx=index)
            for index in range(self.num_envs)
        ]
        self._lights = [None] * self.num_envs

    def _legacy_observations(
        self, observations: list[dict], indices: list[int]
    ) -> list[dict]:
        for observation, index in zip(observations, indices, strict=True):
            robot = self.env.scenes[index].robots[0]
            proprio = observation[robot.name]["proprio"]
            if proprio.shape == (61,):
                # The 2026 demos record the SDK's native robot-frame base velocity.
                continue
            proprio = proprio.clone()
            if proprio.shape != (256,):
                raise ValueError(
                    f"Expected legacy256 proprio, got {tuple(proprio.shape)}."
                )
            # OG 3.9.3 rotates holonomic base velocity into the robot frame.
            # Existing RLinf checkpoints use the old locomotion_robot contract:
            # raw virtual-joint velocities, before this yaw rotation. Adapt ONLY
            # the RLinf observation; leave the simulator's own state untouched.
            proprio[253:256] = robot.get_joint_velocities()[robot.base_control_idx]
            observation[robot.name]["proprio"] = proprio
        return observations

    def _reset_lights(self, indices: list[int]) -> bool:
        if self.activity_name != "turning_out_all_lights_before_sleep":
            return False
        from omnigibson.eval.utils.light_utils import LightToggleSynchronizer

        for index in indices:
            synchronizer = LightToggleSynchronizer(self.env.scenes[index])
            synchronizer.reset_from_current_state()
            self._lights[index] = synchronizer
        return True

    def _rerender(self) -> None:
        # Matches the official reset/light refresh; render only, no physics time.
        for _ in range(3):
            self.og.sim.render()

    def reset(
        self, env_indices: list[int] | torch.Tensor | np.ndarray | None = None
    ) -> tuple[list[dict], list[dict]]:
        """Reset all scenes and copy observations plus per-scene episode info.

        Native reset performs its own initialization/settling physics. A partial
        reset is rejected; use separate simulator processes for isolated scenes.
        Command history is cleared only after reset completes successfully.

        Args:
            env_indices: All scene indices, or None. Empty indices are a no-op.
        """
        self._check_access()
        if env_indices is None:
            indices = list(range(self.num_envs))
        else:
            indices = (
                torch.as_tensor(env_indices, dtype=torch.long).reshape(-1).tolist()
            )
        if len(indices) != len(set(indices)) or any(
            i < 0 or i >= self.num_envs for i in indices
        ):
            raise ValueError(f"Invalid reset indices: {indices}")
        if not indices:
            return [], []
        if set(indices) != set(range(self.num_envs)):
            raise ValueError(
                "Native OmniGibson batches share physics and cannot reset one scene without "
                "advancing others. Use one environment per subprocess for partial reset."
            )
        self._reset_required = True
        selected = self.instances.select(indices)
        needs_rebuild = self.instances.mode == "online" and self._has_reset
        self._has_reset = False
        if selected and self.instances.file_format == "template":
            needs_rebuild = self.config["scene"].get("scene_file") != selected[0].path
            self.config = self.instances.template_config(self.config, selected[0])
        if needs_rebuild:
            # Environment.reload imports another scene without clearing the old
            # stage. The public clear+constructor path is required for templates
            # and fresh online sampling in an isolated simulator process.
            self.og.clear()
            self.env = self.og.Environment(configs=self.config)
            self.og.sim.update_handles()
            self.instances.bind(self.env)
            self._configure_loaded_environment()
        if selected and self.instances.file_format == "tro_state":
            self.instances.load_tro_batch(self.env, selected)
        for index, entry in selected.items():
            self._instance_ids[index] = entry.instance_id
        index_tensor = torch.tensor(indices, dtype=torch.long)
        observations, reset_info = self.env.reset(env_indices=index_tensor)
        if self._reset_lights(indices):
            self._rerender()
            observations, _ = self.env.get_obs(env_indices=index_tensor)
        observations = self._legacy_observations(observations, indices)
        # New native reset returns {'obs_info': list}, old RLinf expects list.
        obs_infos = reset_info.get("obs_info", [{} for _ in indices])
        infos = []
        for offset, index in enumerate(indices):
            self.metrics[index].reset(env=self.env)
            self._frozen[index] = False
            self._terminal[index] = None
            infos.append(
                {
                    "obs_info": obs_infos[offset],
                    "instance_id": self._instance_ids[index],
                    "episode_length": 0,
                }
            )
        self._metrics_snapshot = []
        for index in range(self.num_envs):
            self._metrics_snapshot.append(
                {
                    "official_metrics": {},
                    "q_score": None,
                    "instance_id": self._instance_ids[index],
                    "episode_length": 0,
                }
            )
        self._last_actions.zero_()
        self._last_executed_actions = [None] * self.num_envs
        self._execution_outcome_known = True
        self._reset_required = False
        self._has_reset = True
        return snapshot((observations, infos))

    def step(
        self,
        actions: torch.Tensor | np.ndarray | list[list[float]],
        *,
        render: bool = True,
        get_obs: bool = True,
    ) -> tuple[list[dict] | None, torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
        """Execute one native23 action per scene and copy the Gym five-tuple.

        Args:
            actions: Finite array of shape (num_envs, 23), in native controller
                input units. These are not physical 28-joint target positions.
            render: Refresh images during the step. A skipped render is marked
                ``not_refreshed``; final episode images are always refreshed.
            get_obs: Return a copy of the observations. RLinf skips the copy
                for intermediate chunk steps, while retaining terminal frames.

        Returns:
            Observations, rewards, terminations, truncations and info. Frozen
            scenes return their captured terminal frame and zero reward.

        Raises:
            RuntimeError: Reset is required, including after a failed step.
        """
        self._check_access()
        if not self._has_reset:
            raise RuntimeError("Call reset() before step().")
        if self._reset_required:
            raise RuntimeError("The preceding step failed; call reset() before step().")
        actions = torch.as_tensor(actions, dtype=torch.float32).detach().cpu().clone()
        if actions.shape != (self.num_envs, 23) or not torch.isfinite(actions).all():
            raise ValueError(
                f"Expected finite actions with shape ({self.num_envs}, 23), got {tuple(actions.shape)}."
            )
        active = (~self._frozen).nonzero(as_tuple=True)[0].tolist()
        if not active:
            return self._cached_step(get_obs)
        for index in self._frozen.nonzero(as_tuple=True)[0].tolist():
            actions[index] = self._last_actions[index]
            actions[index, :3] = (
                0  # A zero whole action would command all joints to zero.
            )
        self._reset_required = True
        self._execution_outcome_known = False
        render_context = nullcontext() if render else self.og.sim.render_on_step(False)
        with render_context:
            observations, rewards, terminated, truncated, infos = self.env.step(actions)
        self._last_actions = actions.clone()
        self._last_executed_actions = [row.clone() for row in actions]
        self._execution_outcome_known = True
        terminated = torch.as_tensor(terminated, dtype=torch.bool).cpu()
        truncated = torch.as_tensor(truncated, dtype=torch.bool).cpu()
        rewards = torch.as_tensor(rewards, dtype=torch.float32).cpu()
        newly_finished = truncated | (terminated & (not self.ignore_terminations))
        lights_changed = False
        for index in active:
            if self._lights[index] is not None:
                self._lights[index].sync_from_current_state()
                lights_changed = True
        # A terminal step within a skipped-render chunk still needs a current
        # final image, never the preceding policy-call image.
        refreshed = (render and lights_changed) or (
            not render and any(bool(newly_finished[i]) for i in active)
        )
        if refreshed:
            self._rerender()
            observations, _ = self.env.get_obs()
        observations = self._legacy_observations(
            observations, list(range(self.num_envs))
        )
        for index in active:
            self.metrics[index].step(
                env=self.env,
                action=actions[index],
                obs=observations[index],
                reward=float(rewards[index]),
                terminated=bool(terminated[index]),
                truncated=bool(truncated[index]),
                info=infos[index],
            )
            official_metrics = self.metrics[index].aggregate(env=self.env)
            infos[index]["official_metrics"] = official_metrics
            infos[index]["q_score"] = float(official_metrics["q_score"]["final"])
            infos[index]["instance_id"] = self._instance_ids[index]
            # Native info is populated before the environment counter increment;
            # RLinf episode length counts actions that have actually executed.
            infos[index]["episode_length"] = self.metrics[index].timesteps
            infos[index]["image_freshness"] = (
                "rendered" if render or refreshed else "not_refreshed"
            )
            if newly_finished[index]:
                self._terminal[index] = snapshot(
                    (
                        observations[index],
                        bool(terminated[index]),
                        bool(truncated[index]),
                        infos[index],
                    )
                )
        # An already terminated scene in a shared batch may physically advance;
        # its RLinf terminal observation/score must remain the captured endpoint.
        for index in self._frozen.nonzero(as_tuple=True)[0].tolist():
            observations[index], terminated[index], truncated[index], infos[index] = (
                self._terminal[index]
            )
            rewards[index] = 0.0
        self._frozen |= newly_finished
        self._metrics_snapshot = [
            snapshot(
                {
                    key: info[key]
                    for key in (
                        "official_metrics",
                        "q_score",
                        "instance_id",
                        "episode_length",
                    )
                }
            )
            for info in infos
        ]
        self._reset_required = False
        return snapshot(
            (observations if get_obs else None, rewards, terminated, truncated, infos)
        )

    def _cached_step(self, get_obs: bool):
        return (
            [snapshot(row[0]) for row in self._terminal] if get_obs else None,
            torch.zeros(self.num_envs),
            torch.tensor([row[1] for row in self._terminal]),
            torch.tensor([row[2] for row in self._terminal]),
            [snapshot(row[3]) for row in self._terminal],
        )

    def close(self) -> None:
        """End this dedicated simulator process using the official SDK shutdown.

        Save results and errors and flush logs before calling: Isaac's default
        fast shutdown may terminate Python, so returning is not guaranteed.
        The external process owner must verify the recorded outcome, process
        exit and resource release. Native handles expire immediately; a new
        simulator requires a fresh process. If environment cleanup raises, the
        error propagates without SDK shutdown so the owner can report it.
        """
        if self._closed:
            return
        self._check_access()
        self._closed = True
        if self._owns_simulator:
            self._owns_simulator = False
            if self.env is not None:
                self.env.close()
            # This avoids OG's explicit Python exit when no app exists; it does
            # not change Isaac's default terminal application shutdown behavior.
            self.og.shutdown(due_to_signal=True)
