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


class ActivityInstanceLoader:
    """Select cached instances without changing RLinf's instance-ID meaning."""

    def __init__(self, cfg: DictConfig, num_envs: int, seed: int):
        task = OmegaConf.select(cfg, "omni_config.task")
        self.activity_name = str(task.activity_name)
        self.definition_id = int(task.get("activity_definition_id", 0))
        self.instance_id = int(task.get("activity_instance_id", 0))
        self.mode = str(task.get("instance_resample_mode", "disabled")).lower()
        self.file_format = str(task.get("instance_file_format", "tro_state")).lower()
        self.directory = task.get("activity_instance_dir")
        # Explicit legacy paths take precedence over a preset's default split.
        self.split = None if self.directory else task.get("instance_split")
        self.num_envs = num_envs
        self.random = [random.Random(seed + index) for index in range(num_envs)]
        self.files = ()
        self.default_robot_poses = []
        if self.mode not in SUPPORTED_INSTANCE_RESAMPLE_MODES:
            raise ValueError(f"Invalid instance_resample_mode: {self.mode!r}")
        if self.file_format not in SUPPORTED_INSTANCE_FILE_FORMATS:
            raise ValueError(f"Invalid instance_file_format: {self.file_format!r}")
        if self.split not in (None, "train", "public_test", "hidden_test"):
            raise ValueError(f"Invalid instance_split: {self.split!r}")
        if self.mode == "online":
            if (
                self.directory
                or self.split
                or not task.get("online_object_sampling", False)
            ):
                raise ValueError(
                    "online resampling requires online_object_sampling=true and no cached instance source."
                )
            if task.get("use_presampled_robot_pose", True):
                raise ValueError(
                    "online resampling requires use_presampled_robot_pose=false."
                )
            if num_envs != 1:
                raise ValueError(
                    "OmniGibson 3.9.3 online sampling requires one environment per process."
                )
        elif task.get("online_object_sampling", False) and (
            self.directory or self.split
        ):
            raise ValueError("Cached instances require online_object_sampling=false.")
        if self.directory:
            self.files = tuple(
                discover_activity_instance_files(
                    self.directory,
                    self.activity_name,
                    self.definition_id,
                    self.file_format,
                )
            )
        elif self.split is not None:
            self.files = self._official_files(cfg)
        elif self.mode == "offline":
            raise ValueError(
                "offline resampling requires activity_instance_dir or instance_split."
            )
        if self.files and self.mode == "disabled":
            self._fixed_file()
        if self.files and self.file_format == "template" and num_envs != 1:
            raise ValueError(
                "Template reload requires one environment per process; native batches share one scene config."
            )

    def _official_files(self, cfg: DictConfig) -> tuple[ActivityInstanceFile, ...]:
        from omnigibson.tasks.behavior_task import BehaviorTask
        from omnigibson.utils.asset_utils import get_task_instance_path

        scene = OmegaConf.select(cfg, "omni_config.scene.scene_model")
        name = BehaviorTask.get_cached_activity_scene_filename(
            scene, self.activity_name, self.definition_id, self.instance_id
        )
        suffix = "-tro_state" if self.file_format == "tro_state" else ""
        path = get_task_instance_path(
            scene,
            f"{scene}_task_{self.activity_name}_instances/{name}{suffix}",
            mode=self.split,
        )
        if path is None:
            raise FileNotFoundError(
                f"Official {self.split} instance {self.instance_id} for {self.activity_name!r} "
                "was not found. activity_instance_id is the actual asset ID (public_test starts "
                "at 301); this adapter never silently remaps ID 0 to 301."
            )
        return tuple(
            discover_activity_instance_files(
                Path(path).parent,
                self.activity_name,
                self.definition_id,
                self.file_format,
            )
        )

    def _fixed_file(self) -> ActivityInstanceFile:
        for entry in self.files:
            if entry.instance_id == self.instance_id:
                return entry
        raise ValueError(
            f"Instance ID {self.instance_id} is absent from the configured source."
        )

    def initial_config(self, config: dict) -> dict:
        """A TRO is loaded onto the task's canonical base scene, as in official eval."""
        result = deepcopy(config)
        if self.files:
            if self.file_format == "tro_state":
                result["task"]["activity_instance_id"] = 0
            else:
                entry = self._fixed_file() if self.mode == "disabled" else self.files[0]
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
        if not self.files:
            return {}
        return {
            index: self.random[index].choice(self.files)
            if self.mode == "offline"
            else self._fixed_file()
            for index in env_indices
        }

    def load_tro_batch(self, env, selected: dict[int, ActivityInstanceFile]) -> None:
        """Restore object states, settle once, and save new per-scene reset states.

        Mirrors official BatchedEvaluator's loading order through public scene,
        robot, and BehaviorTask APIs; no evaluator policy or run loop is needed.
        """
        import omnigibson as og
        from omnigibson.tasks.behavior_task import get_presampled_robot_poses
        from omnigibson.utils.bddl_utils import is_system_bddl_inst
        from omnigibson.utils.python_utils import recursively_convert_to_torch

        for index, entry in selected.items():
            scene = env.scenes[index]
            robot = scene.robots[0]
            robot.reset()
            with open(entry.path, encoding="utf-8") as stream:
                states = recursively_convert_to_torch(json.load(stream))
            poses = states.pop("robot_poses", None)
            for name, state in states.items():
                entity = env.task.object_scopes[index].get(name)
                if entity is None:
                    raise ValueError(
                        f"TRO object {name!r} does not exist in environment {index}: {entry.path}"
                    )
                if entity is robot:
                    continue
                if isinstance(state, dict) and isinstance(state.get("root_link"), dict):
                    root = state["root_link"]
                    if "pos" in root and "ori" in root:
                        root["pos"], root["ori"] = (
                            scene.convert_scene_relative_pose_to_world(
                                root["pos"], root["ori"]
                            )
                        )
                entity.load_state(state, serialized=False)
            if poses is None:
                # Official reset requires nonempty pose metadata. Use the clean
                # canonical pose, never stale metadata from the previous TRO.
                position, orientation = self.default_robot_poses[index]
                poses = {
                    "robot": [
                        {
                            "position": position.clone(),
                            "orientation": orientation.clone(),
                        }
                    ]
                }
            pose = get_presampled_robot_poses(
                poses, robot.model, pose_key=env.task.presampled_pose_key
            )[0]
            robot.set_position_orientation(
                pose["position"], pose["orientation"], frame="scene"
            )
            robot.keep_still()
            scene.write_task_metadata(key="robot_poses", data=poses)
            if self.activity_name == "turning_out_all_lights_before_sleep":
                from omnigibson.eval.utils.light_utils import set_light_control_toggles

                set_light_control_toggles(env.task.object_scopes[index].values(), True)
        og.sim.update_handles()
        # Partial resets would also advance the physics of the other scenes.
        for _ in range(25):
            og.sim.step_physics()
            for index in selected:
                for name, entity in env.task.object_scopes[index].items():
                    if entity is not None and not is_system_bddl_inst(name):
                        entity.keep_still()
        for index in selected:
            env.scenes[index].update_initial_file()
