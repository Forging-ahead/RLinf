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
import shlex
import sys
import warnings
from typing import Any

import gymnasium as gym
import torch

from rlinf.envs.sim.behavior.utils import (
    CAMERA_LINKS,
    convert_uint8_rgb,
    validate_simulator_version,
)


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
    """Own one simulator instance in a Ray actor, isolated from other stages."""

    def __init__(self, cfg: Any, num_envs: int, seed: int):
        from .simulator import BehaviorSimulator

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

    def chunk_step(self, actions: torch.Tensor) -> tuple[list, ...]:
        columns: tuple[list, ...] = ([], [], [], [], [])
        for index in range(actions.shape[1]):
            need_obs = not self.skip_obs or index == actions.shape[1] - 1
            result = self.backend.step(actions[:, index], render=need_obs)
            for column, value in zip(columns, result):
                column.append(_cpu(value))
            if not need_obs:
                columns[0][-1] = None
        return columns

    def close(self) -> None:
        self.backend.close()


class BehaviorProcessPool:
    """A private pool for one BehaviorEnv; no simulator spans pipeline stages.

    By default each environment owns an actor so partial reset cannot advance
    the physics of another episode. Explicit batching is available for callers
    that reset complete batches and disable automatic reset.
    """

    def __init__(self, cfg: Any, num_envs: int, seed: int):
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        self.ray = ray
        requested = cfg.get("num_env_subprocess", 1)
        count = num_envs if requested is None else int(requested)
        if not 1 <= count <= num_envs:
            raise ValueError("num_env_subprocess must be null or in [1, num_envs].")
        if cfg.auto_reset and count != num_envs:
            raise ValueError(
                "BEHAVIOR 3.9.3 auto_reset requires one actor per environment. "
                "Set num_env_subprocess: null to select this automatically."
            )
        self.num_envs = num_envs
        rows_per_actor, extra_rows = divmod(num_envs, count)
        self.rows = []
        start = 0
        for index in range(count):
            stop = start + rows_per_actor + (index < extra_rows)
            self.rows.append(list(range(start, stop)))
            start = stop
        self.env_processes: list = []
        # Ray children must retain the environment worker's assigned GPU. They
        # do not reserve new GPUs independently of RLinf's placement system.
        env_vars = {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            env_vars["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
        actor = ray.remote(num_cpus=1)(BehaviorProcess)
        node_id = ray.get_runtime_context().get_node_id()
        try:
            for rows in self.rows:
                self.env_processes.append(
                    actor.options(
                        scheduling_strategy=NodeAffinitySchedulingStrategy(
                            node_id=node_id, soft=False
                        ),
                        runtime_env={
                            "py_executable": shlex.quote(sys.executable),
                            "env_vars": env_vars,
                        },
                    ).remote(cfg, len(rows), seed + rows[0])
                )
            # Wait for all actors to initialize before using the pool.
            names = ray.get(
                [process.get_activity_name.remote() for process in self.env_processes]
            )
            if len(set(names)) != 1:
                raise RuntimeError(f"BEHAVIOR actors loaded different tasks: {names}")
            self.activity_name = names[0]
        except BaseException:
            try:
                self.close()
            except Exception:
                # Preserve initialization errors even if warnings are errors.
                pass
            raise

    def reset(self, env_indices: list[int] | None = None) -> tuple[list, list]:
        """Reset selected rows in caller order, without touching other actors."""
        selected = (
            list(range(self.num_envs)) if env_indices is None else list(env_indices)
        )
        if len(set(selected)) != len(selected) or any(
            index < 0 or index >= self.num_envs for index in selected
        ):
            raise ValueError("reset indices must be unique valid environment indices.")
        plan = []
        for process, rows in zip(self.env_processes, self.rows):
            local = [rows.index(index) for index in selected if index in rows]
            if local:
                # Validate before dispatch so an invalid partial batch does not
                # reset some actors and then fail on another one.
                if len(local) != len(rows):
                    raise ValueError(
                        "Partial reset within a native 3.9.3 batch advances other "
                        "scenes. Use num_env_subprocess: null for independent reset."
                    )
                plan.append((process, rows, local))
        refs = [process.reset.remote(local) for process, _rows, local in plan]
        output: dict[int, tuple[Any, Any]] = {}
        for (_process, rows, local), (observations, infos) in zip(
            plan, self.ray.get(refs)
        ):
            for index, obs, info in zip(local, observations, infos):
                output[rows[index]] = (obs, info)
        return (
            [output[index][0] for index in selected],
            [output[index][1] for index in selected],
        )

    def chunk_step(self, actions: torch.Tensor) -> tuple[list, ...]:
        """Advance every row owned by this stage and restore original row order."""
        refs = [
            process.chunk_step.remote(actions[rows])
            for process, rows in zip(self.env_processes, self.rows)
        ]
        shards = self.ray.get(refs)
        output: tuple[list, ...] = ([], [], [], [], [])
        for time_index in range(actions.shape[1]):
            keep_obs = shards[0][0][time_index] is not None
            obs = [None] * self.num_envs if keep_obs else None
            infos = [None] * self.num_envs
            rewards = torch.zeros(self.num_envs, dtype=torch.float32)
            terms = torch.zeros(self.num_envs, dtype=torch.bool)
            truncs = torch.zeros_like(terms)
            for rows, shard in zip(self.rows, shards):
                for local, row in enumerate(rows):
                    if keep_obs:
                        obs[row] = shard[0][time_index][local]
                    rewards[row] = shard[1][time_index][local]
                    terms[row] = shard[2][time_index][local]
                    truncs[row] = shard[3][time_index][local]
                    infos[row] = shard[4][time_index][local]
            for column, value in zip(output, (obs, rewards, terms, truncs, infos)):
                column.append(value)
        return output

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
            self._ensure_pool()

    def _ensure_pool(self) -> None:
        if self.pool is None:
            # EnvWorker offsets identify stages, not individual environments.
            pool_seed = int(self.cfg.seed) + self.seed_offset * self.num_envs
            pool = BehaviorProcessPool(self.cfg, self.num_envs, pool_seed)
            try:
                self._load_tasks_cfg(pool.activity_name)
            except BaseException:
                try:
                    pool.close()
                except Exception:
                    pass
                raise
            self.pool = pool
            self.pool_offset = 0

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
        self._ensure_pool()
        result = self.pool.reset(env_indices)
        self._requires_reset = False
        return result

    def env_chunk_step(self, chunk_actions: torch.Tensor) -> tuple[list, ...]:
        """Return raw per-step simulator results for one action chunk."""
        if self._requires_reset:
            raise RuntimeError(
                "Call reset() before stepping, including after offload()."
            )
        return self.pool.chunk_step(chunk_actions)

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
        reward_tensor = (
            torch.stack(
                [torch.as_tensor(value, dtype=torch.float32) for value in rewards],
                dim=1,
            )
            * self.reward_coef
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
            pool, self.pool = self.pool, None
            self.pool_offset = None
            pool.close()
