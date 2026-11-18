# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""In-process OmniGibson 3.9.3 environment and RLinf-compatible observations."""

from __future__ import annotations

import random
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from threading import get_ident
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .instance_loader import ActivityInstanceLoader
from .utils import (
    CAMERA_LINKS,
    configure_macros,
    make_environment_config,
    set_head_camera_aperture,
    validate_simulator_version,
)

if TYPE_CHECKING:
    from omnigibson.envs.env_base import Environment
    from omnigibson.robots.robot import Robot


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

    def observe(self, *, render: bool = False) -> tuple[list[dict], list[dict]]:
        """Copy live observations without physics steps or metric updates.

        Args:
            render: Refresh camera annotators by rendering without physics.
                False returns available buffers, whose images may be stale.

        Returns:
            Per-scene observations and info with ``image_freshness`` equal to
            ``rendered`` or ``not_refreshed``. Unlike frozen terminal returns
            from step(), this reads the live simulator, even for frozen scenes.
        """
        self._check_access(require_reset=True)
        if render:
            self._rerender()
        observations, obs_infos = self.env.get_obs()
        observations = self._legacy_observations(
            observations, list(range(self.num_envs))
        )
        infos = [
            {
                "obs_info": info,
                "image_freshness": "rendered" if render else "not_refreshed",
                "state_source": "live_simulator",
                "frozen": bool(self._frozen[index]),
            }
            for index, info in enumerate(obs_infos)
        ]
        return snapshot((observations, infos))

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
        is current, so pair it with observe(render=True) immediately beforehand
        on the same thread; it need not match an older cached image. The SDK may
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
            if proprio.shape != (256,):
                raise ValueError(
                    f"Expected legacy256 proprio, got {tuple(proprio.shape)}."
                )
            # Validate field positions against live robot joint groupings, rather
            # than accepting a 256-vector that happens to have the same size.
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
            proprio = observation[robot.name]["proprio"].clone()
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
    ) -> tuple[list[dict], torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
        """Execute one native23 action per scene and copy the Gym five-tuple.

        Args:
            actions: Finite array of shape (num_envs, 23), in native controller
                input units. These are not physical 28-joint target positions.
            render: Refresh images during the step. A skipped render is marked
                ``not_refreshed``; final episode images are always refreshed.

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
            return self._cached_step()
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
        refreshed = lights_changed or (
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
                snapshot(self._terminal[index])
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
        return snapshot((observations, rewards, terminated, truncated, infos))

    def _cached_step(self):
        rows = [snapshot(row) for row in self._terminal]
        return (
            [row[0] for row in rows],
            torch.zeros(self.num_envs),
            torch.tensor([row[1] for row in rows]),
            torch.tensor([row[2] for row in rows]),
            [row[3] for row in rows],
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
