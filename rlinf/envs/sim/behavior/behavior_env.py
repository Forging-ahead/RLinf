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
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .instance_loader import ActivityInstanceLoader
from .utils import (
    apply_env_wrapper,
    apply_runtime_renderer_settings,
    convert_uint8_rgb,
    setup_omni_cfg,
    validate_simulator_version,
)


def _snapshot(value: Any) -> Any:
    """Copy values that the simulator can mutate on the next physics step."""
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_snapshot(item) for item in value)
    return deepcopy(value)


class BehaviorProcess:
    """Own the native BEHAVIOR environment for one RLinf simulator actor."""

    def __init__(self, cfg: DictConfig, num_envs: int = 1, seed: int = 0) -> None:
        validate_simulator_version(cfg)
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
                "BEHAVIOR requires OmniGibson 3.9.3 (v3.9.3-post2 API); "
                f"found {getattr(og, '__version__', 'unknown')!r}."
            )
        self.og = og
        omni_cfg = setup_omni_cfg(cfg, self.num_envs)
        self.instances = ActivityInstanceLoader.from_omni_cfg(
            cfg.omni_config, num_envs=self.num_envs, seed=seed
        )
        self.activity_name = self.instances.activity_name
        self.config = self.instances.initial_config(
            OmegaConf.to_container(omni_cfg, resolve=True)
        )
        self.ignore_terminations = bool(
            OmegaConf.select(cfg, "ignore_terminations", default=False)
        )
        self._has_reset = False
        self._frozen = torch.zeros(self.num_envs, dtype=torch.bool)
        self._terminal = [None] * self.num_envs
        self._last_actions = torch.zeros((self.num_envs, 23))
        self._instance_ids = [self.instances.activity_instance_id] * self.num_envs
        self.env = og.Environment(configs=self.config)
        og.sim.update_handles()
        self.instances.bind(self.env)
        self._configure_loaded_environment()

        # Kit disables cyclic GC at startup. Re-enable it for long RLinf runs.
        gc.enable()

    def get_activity_name(self):
        return self.activity_name

    def _call_step(self, actions, render=True):
        context = nullcontext() if render else self.og.sim.render_on_step(False)
        with context:
            return self.env.step(actions)

    def _call_reset(self, reset_indices):
        indices = torch.tensor(reset_indices, dtype=torch.long)
        return self.env.reset(env_indices=indices)

    def _step_shard(self, actions, need_obs):
        """Advance a native batch and keep each finished scene's terminal result."""
        if not self._has_reset:
            raise RuntimeError("Call reset() before stepping the simulator.")
        actions = torch.as_tensor(actions, dtype=torch.float32).detach().cpu().clone()
        if actions.shape != (self.num_envs, 23) or not torch.isfinite(actions).all():
            raise ValueError(
                f"Expected finite actions with shape ({self.num_envs}, 23), got {tuple(actions.shape)}."
            )
        active = (~self._frozen).nonzero(as_tuple=True)[0].tolist()
        if not active:
            return self._cached_step(need_obs)
        for index in self._frozen.nonzero(as_tuple=True)[0].tolist():
            actions[index] = self._last_actions[index]
            actions[index, :3] = (
                0  # A zero whole action would command all joints to zero.
            )
        observations, rewards, terminated, truncated, infos = self._call_step(
            actions, render=need_obs
        )
        self._last_actions = actions
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
        refreshed = (need_obs and lights_changed) or (
            not need_obs and any(bool(newly_finished[i]) for i in active)
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
            if newly_finished[index]:
                self._terminal[index] = _snapshot(
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
        return _snapshot(
            (
                observations if need_obs else None,
                rewards,
                terminated,
                truncated,
                infos,
            )
        )

    def _cached_step(self, need_obs):
        return (
            [_snapshot(row[0]) for row in self._terminal] if need_obs else None,
            torch.zeros(self.num_envs),
            torch.tensor([row[1] for row in self._terminal]),
            torch.tensor([row[2] for row in self._terminal]),
            [_snapshot(row[3]) for row in self._terminal],
        )

    def chunk_step(self, actions):
        skip_obs = self.cfg.get(
            "skip_intermediate_renders",
            self.cfg.get("skip_intermediate_obs_in_chunk", False),
        )
        results = []
        for t in range(actions.shape[1]):
            need_obs = not skip_obs or t == actions.shape[1] - 1
            results.append(self._step_shard(actions[:, t], need_obs))
        return tuple(zip(*results))

    def _configure_loaded_environment(self) -> None:
        from omnigibson.eval.utils.score_utils import load_human_stats
        from omnigibson.metrics.task_metric import TaskMetric

        apply_runtime_renderer_settings()
        wrapper = OmegaConf.select(
            self.cfg, "omni_config.env.env_wrapper", default="rgb"
        )
        self.env = apply_env_wrapper(self.env, wrapper)
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

        Args:
            env_indices: All scene indices, or None. Empty indices are a no-op.
        """
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
        selected = self.instances.select(indices)
        needs_rebuild = (
            self.instances.instance_resample_mode == "online" and self._has_reset
        )
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
        observations, reset_info = self._call_reset(indices)
        if self._reset_lights(indices):
            self._rerender()
            observations, _ = self.env.get_obs(
                env_indices=torch.tensor(indices, dtype=torch.long)
            )
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
        self._last_actions.zero_()
        self._has_reset = True
        return _snapshot((observations, infos))

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
            self.og.shutdown(due_to_signal=True)


class BehaviorProcessPool:
    """Share one pool among the pipeline stages of a worker's train or eval config."""

    _shared_pools: ClassVar[dict[tuple[int, int], BehaviorProcessPool]] = {}

    @classmethod
    def acquire_shared(
        cls, cfg, worker_info, pipeline_stage_num, num_envs, seed_offset
    ):
        """Attach to the shared pool and return ``(pool, pool_offset)``."""
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
        """Drop this stage; tear down the pool when the last stage releases."""
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
        # Create subprocess actors with a retry/backoff loop. Actor startup
        # can fail (e.g. simulator plugin errors); retry a few times to handle
        # transient failures. Configurable via `behavior.init_retry_*` keys.
        startup = cfg.get("behavior", {})
        attempts = max(1, int(startup.get("init_retry_count", 3)))
        delay = float(startup.get("init_retry_delay", 5.0))
        for attempt in range(1, attempts + 1):
            try:
                for rows in self.rows:
                    self.env_processes.append(
                        actor.remote(cfg, len(rows), seed + rows.start)
                    )
                # Wait for all instances to initialize and fetch their activity name
                activity_names_refs = [
                    proc.get_activity_name.remote() for proc in self.env_processes
                ]
                activity_names = ray.get(activity_names_refs)
                if len(set(activity_names)) != 1:
                    raise RuntimeError(
                        f"BEHAVIOR actors loaded different tasks: {activity_names}"
                    )
                self.activity_name = activity_names[0]
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

    def _slice_plan(
        self, global_start: int, num_envs: int, env_indices: list[int] | None = None
    ) -> list[tuple[int, list[int], list[int]]]:
        """Build the per-subprocess plan for a contiguous global slice.

        Returns entries of ``(subproc_idx, slice_positions, local_rows)``.
        ``slice_positions`` are indices inside the caller's slice and
        ``local_rows`` are the matching rows owned by that subprocess.
        """
        indices = range(num_envs) if env_indices is None else env_indices
        if env_indices is not None and (
            len(set(indices)) != len(indices)
            or any(i < 0 or i >= num_envs for i in indices)
        ):
            raise ValueError("reset indices must be unique valid environment indices.")
        positions = {global_start + index: pos for pos, index in enumerate(indices)}
        plan = []
        for sp, rows in enumerate(self.rows):
            local_rows = [i for i, row in enumerate(rows) if row in positions]
            if not local_rows:
                continue
            if len(local_rows) != len(rows):
                raise ValueError(
                    "Partial reset within a native 3.9.3 batch advances other "
                    "scenes. Use num_env_subprocess: null for independent reset."
                )
            plan.append((sp, [positions[rows[i]] for i in local_rows], local_rows))
        return plan

    def env_reset_slice(
        self, global_start: int, num_envs: int, env_indices: list[int] | None = None
    ):
        """Reset envs in ``[global_start, global_start + num_envs)``."""
        plan = self._slice_plan(global_start, num_envs, env_indices)
        if env_indices is not None:
            num_envs = len(env_indices)
        refs = [
            self.env_processes[sp].reset.remote(local_rows)
            for sp, _positions, local_rows in plan
        ]

        shard_results = self.ray.get(refs)
        all_raw_obs: list = [None] * num_envs
        all_infos: list = [None] * num_envs
        for (raw_obs, infos), (_sp, positions, _local_rows) in zip(shard_results, plan):
            for pos, obs, info in zip(positions, raw_obs, infos):
                all_raw_obs[pos] = obs
                all_infos[pos] = info
        return all_raw_obs, all_infos

    def env_chunk_step_slice(
        self,
        global_start: int,
        slice_num_envs: int,
        chunk_actions: torch.Tensor,
    ):
        """Run chunk_step on shards; pool handles all sharding/merging.
        ``chunk_actions`` must be ``[slice_num_envs, chunk, action_dim]``.
        """
        chunk_size = chunk_actions.shape[1]
        plan = self._slice_plan(global_start, slice_num_envs)

        refs = [
            self.env_processes[sp].chunk_step.remote(chunk_actions[positions])
            for sp, positions, _local_rows in plan
        ]
        shard_results = self.ray.get(refs)
        return self._merge_shards(shard_results, plan, slice_num_envs, chunk_size)

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
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        validate_simulator_version(cfg)
        self.cfg = cfg
        self.reward_coef = cfg.get("reward_coef", 1)

        self.num_envs = num_envs
        self.ignore_terminations = cfg.ignore_terminations
        self.seed_offset = seed_offset
        self.seed = self.cfg.seed + seed_offset
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

    def _ensure_pool(self):
        if self.pool is None:
            self.pool, self.pool_offset = BehaviorProcessPool.acquire_shared(
                self.cfg,
                self.worker_info,
                self.pipeline_stage_num,
                self.num_envs,
                self.seed_offset,
            )

    def _load_tasks_cfg(self, activity_name: str):
        # Read task description

        task_description_path = os.path.join(
            os.path.dirname(__file__), "behavior_task.jsonl"
        )
        with open(task_description_path, "r") as f:
            text = f.read()
            task_description = [json.loads(x) for x in text.strip().split("\n") if x]
        task_description_map = {
            task_description[i]["task_name"]: task_description[i]["task"]
            for i in range(len(task_description))
        }
        self.task_description = task_description_map[activity_name]

    def _init_env(self) -> None:
        self._ensure_pool()
        try:
            self._load_tasks_cfg(self.pool.activity_name)
        except BaseException:
            self.close()
            raise

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

    def _extract_obs_image(self, raw_obs):
        state = left_image = right_image = zed_image = None
        for sensor_data in raw_obs.values():
            if not isinstance(sensor_data, dict) or "proprio" not in sensor_data:
                continue
            for k, v in sensor_data.items():
                if "left_realsense_link:Camera:0" in k:
                    left_image = convert_uint8_rgb(v["rgb"])
                elif "right_realsense_link:Camera:0" in k:
                    right_image = convert_uint8_rgb(v["rgb"])
                elif "zed_link:Camera:0" in k:
                    zed_image = convert_uint8_rgb(v["rgb"])
                elif "proprio" in k:
                    state = torch.as_tensor(v)
        if state is None or state.shape not in ((61,), (256,)):
            raise ValueError("BEHAVIOR requires 61 or 256 R1Pro state values.")
        if left_image is None or right_image is None or zed_image is None:
            raise ValueError("BEHAVIOR requires head and both wrist RGB observations.")

        return {
            "main_images": zed_image,  # [H, W, C]
            "wrist_images": torch.stack(
                [left_image, right_image], axis=0
            ),  # [N_IMG, H, W, C]
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
            "task_descriptions": [self.task_description for _ in range(len(obs_list))],
            "states": torch.stack(
                [obs["state"] for obs in extracted_obs_list], axis=0
            ),  # [N_ENV, state_dim]
        }
        return obs

    def _calc_step_reward(self, reward):
        return self.reward_coef * reward

    def reset(self):
        raw_obs, infos = self.env_reset()
        obs = self._wrap_obs(raw_obs)
        self._reset_metrics()
        rewards = torch.zeros(self.num_envs)
        infos = self._record_metrics(rewards, infos)
        return obs, infos

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim].
        chunk_actions = torch.as_tensor(chunk_actions).detach().cpu()
        if (
            chunk_actions.ndim != 3
            or chunk_actions.shape[0] != self.num_envs
            or chunk_actions.shape[1] < 1
            or chunk_actions.shape[2] != 23
        ):
            raise ValueError(
                f"Expected actions [{self.num_envs}, T>=1, 23], got {tuple(chunk_actions.shape)}."
            )
        if not torch.isfinite(chunk_actions).all():
            raise ValueError("BEHAVIOR actions must be finite.")
        (
            raw_obs_list,
            raw_rewards_list,
            raw_terminations_list,
            raw_truncations_list,
            raw_infos_list,
        ) = self.env_chunk_step(chunk_actions)

        obs_list = []
        infos_list = []
        scaled_rewards_list = []
        merged_terminations_list = []
        for raw_obs, raw_rewards, raw_terminations, step_infos in zip(
            raw_obs_list,
            raw_rewards_list,
            raw_terminations_list,
            raw_infos_list,
        ):
            if raw_obs is None:
                obs_list.append(None)
            else:
                obs_list.append(self._wrap_obs(raw_obs))
            step_rewards = self._calc_step_reward(raw_rewards)
            self._steps = torch.tensor(
                [info["episode_length"] for info in step_infos], device=self.device
            )
            infos_list.append(self._record_metrics(step_rewards, step_infos))
            if self.ignore_terminations:
                raw_terminations = torch.zeros_like(raw_terminations)
            merged_terminations_list.append(raw_terminations)
            scaled_rewards_list.append(step_rewards)
        if obs_list[-1] is None:
            raise RuntimeError(
                "BEHAVIOR must return the final observation of every chunk."
            )

        chunk_rewards = torch.stack(
            scaled_rewards_list, dim=1
        )  # [num_envs, chunk_steps]
        raw_terminations = torch.stack(
            merged_terminations_list, dim=1
        )  # [num_envs, chunk_steps]
        raw_truncations = torch.stack(
            raw_truncations_list, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_terminations.any(dim=1)
        past_truncations = raw_truncations.any(dim=1)

        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )

        chunk_terminations = torch.zeros_like(raw_terminations)
        chunk_terminations[:, -1] = past_terminations

        chunk_truncations = torch.zeros_like(raw_truncations)
        chunk_truncations[:, -1] = past_truncations
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    @property
    def device(self) -> str:
        return self._device

    @property
    def elapsed_steps(self) -> torch.Tensor:
        return self._steps.clone()

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        if env_idx is None:
            env_idx = slice(None)
        self._steps[env_idx] = 0
        if not self.record_metrics:
            return
        self.success_once[env_idx] = False
        self.returns[env_idx] = 0

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
        lengths = self.elapsed_steps
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

    def _handle_auto_reset(self, dones, final_obs, final_info):
        indices = dones.nonzero(as_tuple=False).flatten().tolist()
        raw_obs, _ = self.env_reset(indices)
        reset_obs = self._wrap_obs(raw_obs)
        # Replace completed rows only. Keep the terminal observations for bootstrapping.
        extracted_obs = {}
        for key, value in final_obs.items():
            merged = value.clone() if isinstance(value, torch.Tensor) else list(value)
            for local, index in enumerate(indices):
                merged[index] = reset_obs[key][local]
            extracted_obs[key] = merged
        self._reset_metrics(indices)
        infos = {}
        if self.record_metrics:
            infos["episode"] = {}
            for key, value in final_info["episode"].items():
                infos["episode"][key] = value.clone()
                infos["episode"][key][indices] = 0
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    def update_reset_state_ids(self):
        # use for multi task training
        pass

    def offload(self):
        self.close()

    def close(self):
        self._requires_reset = True
        if self.pool:
            self.pool.release_shared(self.pool_offset)
            self.pool = None
            self.pool_offset = None
