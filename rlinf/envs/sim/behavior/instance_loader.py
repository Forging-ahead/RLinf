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

import json
import os
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

TASK_INSTANCE_FILE_SUFFIX = "_template-tro_state.json"
TASK_INSTANCE_TEMPLATE_FILE_SUFFIX = "_template.json"
SUPPORTED_INSTANCE_RESAMPLE_MODES = ("disabled", "offline", "online")
SUPPORTED_INSTANCE_FILE_FORMATS = ("template", "tro_state")


@dataclass(frozen=True)
class ActivityInstanceFile:
    """Metadata for one cached BEHAVIOR activity instance file."""

    instance_id: int
    path: str
    file_format: str


def parse_activity_instance_filename(
    filename: str,
    activity_name: str,
    instance_file_format: str,
) -> tuple[int, int] | None:
    """Parse definition and instance ids from a cached instance filename.

    Args:
        filename: Candidate cached instance filename.
        activity_name: Expected BEHAVIOR activity name.
        instance_file_format: Expected cached file format. Must be ``template`` or
            ``tro_state``.

    Returns:
        A ``(definition_id, instance_id)`` tuple when the filename matches the
        expected activity and format. Returns ``None`` when it does not match.

    Raises:
        ValueError: If ``instance_file_format`` is unsupported.
    """
    if instance_file_format == "template":
        suffix = TASK_INSTANCE_TEMPLATE_FILE_SUFFIX
    elif instance_file_format == "tro_state":
        suffix = TASK_INSTANCE_FILE_SUFFIX
    else:
        raise ValueError(
            f"Unsupported cached instance format: {instance_file_format!r}."
        )

    if not filename.endswith(suffix):
        return None

    infix = f"_task_{activity_name}_"
    if infix not in filename:
        return None

    stem = filename[: -len(suffix)]
    _, suffix_stem = stem.split(infix, 1)
    definition_and_instance = suffix_stem.split("_")
    if len(definition_and_instance) != 2:
        return None

    definition_id, instance_id = definition_and_instance
    if not definition_id.isdigit() or not instance_id.isdigit():
        return None

    return int(definition_id), int(instance_id)


def discover_activity_instance_files(
    activity_instance_dir: str | os.PathLike[str],
    activity_name: str,
    activity_definition_id: int,
    instance_file_format: str,
) -> list[ActivityInstanceFile]:
    """Discover cached instance files for one BEHAVIOR activity.

    Args:
        activity_instance_dir: Directory containing cached instance JSON files.
        activity_name: Expected BEHAVIOR activity name.
        activity_definition_id: Expected BEHAVIOR activity definition id.
        instance_file_format: Cached instance file format to load.

    Returns:
        A sorted list of discovered cached instance files.

    Raises:
        ValueError: If the directory does not exist, contains duplicate instance
            ids for the requested format, or contains no matching files.
    """
    instance_dir = Path(activity_instance_dir)
    if not instance_dir.is_dir():
        raise ValueError(
            f"activity_instance_dir must be an existing directory, got: {instance_dir}"
        )

    instance_files = {}
    for entry in instance_dir.iterdir():
        if not entry.is_file():
            continue

        parsed = parse_activity_instance_filename(
            entry.name,
            activity_name=activity_name,
            instance_file_format=instance_file_format,
        )
        if parsed is None:
            continue

        definition_id, instance_id = parsed
        if definition_id != activity_definition_id:
            continue
        if instance_id in instance_files:
            raise ValueError(
                "Duplicate activity instance id "
                f"{instance_id} found in {instance_dir} for format "
                f"{instance_file_format!r}."
            )

        instance_files[instance_id] = ActivityInstanceFile(
            instance_id=instance_id,
            path=str(entry),
            file_format=instance_file_format,
        )

    if not instance_files:
        raise ValueError(
            "No cached BEHAVIOR task instances were found in "
            f"{instance_dir} for activity_name={activity_name}, "
            f"activity_definition_id={activity_definition_id}, "
            f"instance_file_format={instance_file_format!r}."
        )

    return [instance_files[k] for k in sorted(instance_files)]


def load_activity_instance_tro_state(
    env,
    instance_id: int,
    tro_file_path: str,
    reset_scene: bool = False,
    *,
    env_index: int = 0,
    default_robot_pose=None,
    settle: bool = True,
) -> None:
    """Apply a cached tro_state file to an existing OmniGibson env.

    Args:
        env: OmniGibson environment to mutate in place.
        instance_id: Activity instance id represented by ``tro_file_path``.
        tro_file_path: Path to a ``*_template-tro_state.json`` file.
        reset_scene: Whether to call ``env.scene.reset()`` after applying the
            cached state.
        env_index: Native scene index to update.
        default_robot_pose: Canonical scene-frame pose for files without robot metadata.
        settle: Defer to the caller when settling several scenes together.
    """
    from omnigibson.tasks.behavior_task import get_presampled_robot_poses
    from omnigibson.utils.python_utils import recursively_convert_to_torch

    scene = env.scenes[env_index]
    robot = scene.robots[0]
    robot.reset()
    with open(tro_file_path, "r", encoding="utf-8") as f:
        tro_state = recursively_convert_to_torch(json.load(f))

    robot_poses = tro_state.pop("robot_poses", None)

    for tro_key, state in tro_state.items():
        entity = env.task.object_scopes[env_index].get(tro_key)
        assert entity is not None, (
            f"Cached task-relevant object {tro_key!r} is not present in the current "
            f"scene {env_index} while loading instance {instance_id} from {tro_file_path}."
        )
        if entity is robot:
            continue
        if (
            scene.idx != 0
            and isinstance(state, dict)
            and isinstance(state.get("root_link"), dict)
            and "pos" in state["root_link"]
            and "ori" in state["root_link"]
        ):
            rebased_state = dict(state)
            rebased_root_link = dict(state["root_link"])
            rebased_pos, rebased_ori = scene.convert_scene_relative_pose_to_world(
                rebased_root_link["pos"],
                rebased_root_link["ori"],
            )
            rebased_root_link["pos"] = rebased_pos
            rebased_root_link["ori"] = rebased_ori
            rebased_state["root_link"] = rebased_root_link
            state = rebased_state
        entity.load_state(state, serialized=False)

    if robot_poses is None:
        # The 2026 task reset requires nonempty pose metadata.
        if default_robot_pose is None:
            default_robot_pose = robot.get_position_orientation(frame="scene")
        position, orientation = default_robot_pose
        robot_poses = {"robot": [{"position": position, "orientation": orientation}]}
    robot_pose = get_presampled_robot_poses(
        robot_poses, robot.model, pose_key=env.task.presampled_pose_key
    )[0]
    robot.set_position_orientation(
        robot_pose["position"],
        robot_pose["orientation"],
        frame="scene",
    )
    robot.keep_still()
    scene.write_task_metadata(key="robot_poses", data=robot_poses)
    if env.task.activity_name == "turning_out_all_lights_before_sleep":
        from omnigibson.eval.utils.light_utils import set_light_control_toggles

        set_light_control_toggles(env.task.object_scopes[env_index].values(), True)

    if settle:
        _settle_tro_states(env, [env_index])
    if reset_scene:
        scene.reset()


def _settle_tro_states(env, indices) -> None:
    """Physics advances all native scenes, so settle once per batch."""
    import omnigibson as og
    from omnigibson.utils.bddl_utils import is_system_bddl_inst

    og.sim.update_handles()
    for _ in range(25):
        og.sim.step_physics()
        for index in indices:
            for name, entity in env.task.object_scopes[index].items():
                if entity is not None and not is_system_bddl_inst(name):
                    entity.keep_still()

    for index in indices:
        env.scenes[index].update_initial_file()


class ActivityInstanceLoader:
    """Prepare BEHAVIOR reset-time task instances for one vectorized env."""

    def __init__(
        self,
        omni_cfg: DictConfig,
        activity_name: str,
        activity_instance_id: int,
        instance_resample_mode: str,
        activity_instances: tuple[ActivityInstanceFile, ...],
        num_envs: int = 1,
        seed: int = 0,
    ):
        self.omni_cfg = omni_cfg
        self.activity_name = activity_name
        self.activity_instance_id = activity_instance_id
        self.instance_resample_mode = instance_resample_mode
        self.activity_instances = activity_instances
        self.file_format = (
            omni_cfg.task.get("instance_file_format") or "tro_state"
        ).lower()
        self.split = (
            None
            if omni_cfg.task.get("activity_instance_dir")
            else omni_cfg.task.get("instance_split")
        )
        self.random = [random.Random(seed + index) for index in range(num_envs)]
        self.default_robot_poses = []

    @classmethod
    def from_omni_cfg(
        cls, omni_cfg: DictConfig, num_envs: int = 1, seed: int = 0
    ) -> "ActivityInstanceLoader":
        """Build an instance loader from OmniGibson task config.

        Args:
            omni_cfg: Full OmniGibson config used to construct the BEHAVIOR env.
            num_envs: Number of native scenes in this process.
            seed: Base seed for per-scene instance sampling.

        Returns:
            A configured activity instance loader.

        Raises:
            ValueError: If the instance-resample configuration is invalid.
        """
        activity_name = OmegaConf.select(omni_cfg, "task.activity_name")
        activity_definition_id = OmegaConf.select(
            omni_cfg, "task.activity_definition_id", default=0
        )
        activity_instance_id = OmegaConf.select(
            omni_cfg, "task.activity_instance_id", default=0
        )
        activity_instance_dir = OmegaConf.select(omni_cfg, "task.activity_instance_dir")
        instance_resample_mode = OmegaConf.select(
            omni_cfg, "task.instance_resample_mode", default="disabled"
        )
        instance_file_format = OmegaConf.select(
            omni_cfg, "task.instance_file_format", default="tro_state"
        )
        online_object_sampling = OmegaConf.select(
            omni_cfg, "task.online_object_sampling"
        )
        use_presampled_robot_pose = OmegaConf.select(
            omni_cfg, "task.use_presampled_robot_pose", default=True
        )

        if not isinstance(instance_resample_mode, str):
            raise ValueError(
                f"task.instance_resample_mode must be a string, got {instance_resample_mode!r}."
            )
        instance_resample_mode = instance_resample_mode.lower()
        if instance_resample_mode not in SUPPORTED_INSTANCE_RESAMPLE_MODES:
            raise ValueError(
                "task.instance_resample_mode must be one of "
                f"{SUPPORTED_INSTANCE_RESAMPLE_MODES}, got {instance_resample_mode!r}."
            )

        if instance_file_format is not None:
            if not isinstance(instance_file_format, str):
                raise ValueError(
                    f"task.instance_file_format must be a string, got {instance_file_format!r}."
                )
            instance_file_format = instance_file_format.lower()
            if instance_file_format not in SUPPORTED_INSTANCE_FILE_FORMATS:
                raise ValueError(
                    "task.instance_file_format must be one of "
                    f"{SUPPORTED_INSTANCE_FILE_FORMATS}, got {instance_file_format!r}."
                )

        split = None if activity_instance_dir else omni_cfg.task.get("instance_split")
        if split not in (None, "train", "public_test", "hidden_test"):
            raise ValueError(f"Invalid instance_split: {split!r}")
        if instance_resample_mode == "online":
            if num_envs != 1:
                raise ValueError(
                    "Online sampling requires one environment per process."
                )
            if activity_instance_dir is not None or split:
                raise ValueError(
                    "task.activity_instance_dir is incompatible with "
                    "task.instance_resample_mode='online'."
                )
            if not online_object_sampling:
                raise ValueError(
                    "task.instance_resample_mode='online' requires "
                    "task.online_object_sampling to be True."
                )
            if use_presampled_robot_pose:
                raise ValueError(
                    "task.instance_resample_mode='online' requires "
                    "task.use_presampled_robot_pose to be False."
                )
        if split is not None:
            # Resolve the named split through the SDK before scanning the original cache layout.
            from omnigibson.tasks.behavior_task import BehaviorTask
            from omnigibson.utils.asset_utils import get_task_instance_path

            scene = omni_cfg.scene.scene_model
            name = BehaviorTask.get_cached_activity_scene_filename(
                scene, activity_name, activity_definition_id, activity_instance_id
            )
            suffix = "-tro_state" if instance_file_format == "tro_state" else ""
            path = get_task_instance_path(
                scene,
                f"{scene}_task_{activity_name}_instances/{name}{suffix}",
                mode=split,
            )
            if path is None:
                raise FileNotFoundError(
                    f"Official {split} instance {activity_instance_id} for {activity_name!r} was not found."
                )
            activity_instance_dir = Path(path).parent

        if activity_instance_dir is None:
            if instance_resample_mode == "offline":
                raise ValueError(
                    "task.activity_instance_dir or task.instance_split must be set when "
                    "task.instance_resample_mode is 'offline'."
                )
            return cls(
                omni_cfg=omni_cfg,
                activity_name=activity_name,
                activity_instance_id=activity_instance_id,
                instance_resample_mode=instance_resample_mode,
                activity_instances=(),
                num_envs=num_envs,
                seed=seed,
            )

        if online_object_sampling:
            raise ValueError(
                "task.activity_instance_dir only supports cached offline instances. "
                "Please disable task.online_object_sampling."
            )
        if instance_file_format is None:
            raise ValueError(
                "task.instance_file_format must be set to 'template' or "
                "'tro_state' when task.activity_instance_dir is set."
            )

        if instance_file_format == "template" and num_envs != 1:
            raise ValueError("Template reload requires one environment per process.")

        activity_instances = tuple(
            discover_activity_instance_files(
                activity_instance_dir=activity_instance_dir,
                activity_name=activity_name,
                activity_definition_id=activity_definition_id,
                instance_file_format=instance_file_format,
            )
        )
        if instance_resample_mode == "disabled":
            instance_ids = {entry.instance_id for entry in activity_instances}
            if activity_instance_id not in instance_ids:
                raise ValueError(
                    f"task.activity_instance_id={activity_instance_id} is not present in "
                    f"task.activity_instance_dir={activity_instance_dir}."
                )

        return cls(
            omni_cfg=omni_cfg,
            activity_name=activity_name,
            activity_instance_id=activity_instance_id,
            instance_resample_mode=instance_resample_mode,
            activity_instances=activity_instances,
            num_envs=num_envs,
            seed=seed,
        )

    def _get_activity_instance(self, instance_id: int) -> ActivityInstanceFile:
        for instance_file in self.activity_instances:
            if instance_file.instance_id == instance_id:
                return instance_file
        raise ValueError(f"Activity instance id {instance_id} was not discovered.")

    def initial_config(self, config: dict) -> dict:
        """A TRO is loaded onto the task's canonical base scene, as in official eval."""
        result = deepcopy(config)
        if self.activity_instances:
            if self.file_format == "tro_state":
                result["task"]["activity_instance_id"] = 0
            else:
                entry = (
                    self._get_activity_instance(self.activity_instance_id)
                    if self.instance_resample_mode == "disabled"
                    else self.activity_instances[0]
                )
                result = self.template_config(result, entry)
        return result

    @staticmethod
    def template_config(config: dict, entry: ActivityInstanceFile) -> dict:
        result = deepcopy(config)
        result["task"]["activity_instance_id"] = entry.instance_id
        result["scene"]["scene_file"] = entry.path
        result["scene"]["scene_instance"] = None
        return result

    def bind(self, env) -> None:
        self.default_robot_poses = [
            tuple(
                value.clone()
                for value in scene.robots[0].get_position_orientation(frame="scene")
            )
            for scene in env.scenes
        ]

    def select(self, env_indices: list[int]) -> dict[int, ActivityInstanceFile]:
        if not self.activity_instances:
            return {}
        return {
            index: self.random[index].choice(self.activity_instances)
            if self.instance_resample_mode == "offline"
            else self._get_activity_instance(self.activity_instance_id)
            for index in env_indices
        }

    def load_tro_batch(self, env, selected: dict[int, ActivityInstanceFile]) -> None:
        for index, instance_file in selected.items():
            load_activity_instance_tro_state(
                env,
                instance_id=instance_file.instance_id,
                tro_file_path=instance_file.path,
                env_index=index,
                default_robot_pose=self.default_robot_poses[index],
                settle=False,
            )
        _settle_tro_states(env, selected)
