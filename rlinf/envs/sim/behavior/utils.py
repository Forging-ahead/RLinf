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

from copy import deepcopy
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

SUPPORTED_ENV_WRAPPERS = ("rgb", "default", "rgb_lowres", "rich_obs")

R1PRO_PROPRIO_KEYS = [
    "joint_qpos",
    "joint_qpos_sin",
    "joint_qpos_cos",
    "joint_qvel",
    "joint_qeffort",
    "robot_pos",
    "robot_ori_cos",
    "robot_ori_sin",
    "robot_2d_ori",
    "robot_2d_ori_cos",
    "robot_2d_ori_sin",
    "robot_lin_vel",
    "robot_ang_vel",
    "arm_left_qpos",
    "arm_left_qpos_sin",
    "arm_left_qpos_cos",
    "arm_left_qvel",
    "eef_left_pos",
    "eef_left_quat",
    "gripper_left_qpos",
    "gripper_left_qvel",
    "arm_right_qpos",
    "arm_right_qpos_sin",
    "arm_right_qpos_cos",
    "arm_right_qvel",
    "eef_right_pos",
    "eef_right_quat",
    "gripper_right_qpos",
    "gripper_right_qvel",
    "trunk_qpos",
    "trunk_qvel",
    "base_qpos",
    "base_qpos_sin",
    "base_qpos_cos",
    "base_qvel",
]


def convert_uint8_rgb(image: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(image):
        image = torch.as_tensor(image)

    if image.dtype == torch.uint8:
        return image[..., :3]

    if torch.is_floating_point(image):
        max_val = float(image.detach().max().item()) if image.numel() > 0 else 1.0
        if max_val <= 1.0 + 1e-6:
            image = image * 255.0
        image = image.round().clamp(0, 255).to(torch.uint8)
    else:
        image = image.clamp(0, 255).to(torch.uint8)

    return image[..., :3]


INSTANCE_KEYS = (
    "activity_instance_dir",
    "instance_resample_mode",
    "instance_file_format",
    "instance_split",
)
CAMERA_LINKS = {
    "head": "zed_link:Camera:0",
    "left_wrist": "left_realsense_link:Camera:0",
    "right_wrist": "right_realsense_link:Camera:0",
}


RGB_RESOLUTIONS = {
    "head": (720, 720),
    "left_wrist": (480, 480),
    "right_wrist": (480, 480),
}


def set_head_camera_aperture(robot) -> None:
    """Apply the R1Pro head camera optics used by the model."""
    name = f"{robot.name}:{CAMERA_LINKS['head']}"
    if name not in robot.sensors:
        raise ValueError(f"R1Pro head camera is absent: {name}")
    robot.sensors[name].horizontal_aperture = 40.0


def make_environment_config(
    cfg: DictConfig, num_envs: int, omnigibson_path: str
) -> dict:
    """Keep RLinf overrides, with canonical 3.9.3 robot and camera settings.

    The official evaluation driver is deliberately not constructed here: it
    unconditionally chooses its own scene, rates, robot mass, and camera profile.
    """
    base_name = OmegaConf.select(cfg, "base_config_name", default="r1pro_behavior")
    base = OmegaConf.load(str(Path(omnigibson_path) / "configs" / f"{base_name}.yaml"))
    overrides = OmegaConf.select(cfg, "omni_config")
    if overrides is None:
        raise ValueError("BEHAVIOR requires omni_config.")
    # OmegaConf replaces lists. Merge the single robot separately so controller
    # overrides do not accidentally discard the remaining robot configuration.
    base_dict = OmegaConf.to_container(base, resolve=True)
    override_dict = OmegaConf.to_container(overrides, resolve=True)
    robots = override_dict.pop("robots", None)
    if robots is None or len(robots) != 1:
        raise ValueError("BEHAVIOR requires exactly one omni_config.robots entry.")
    robot = OmegaConf.to_container(
        OmegaConf.merge(base_dict["robots"][0], robots[0]), resolve=True
    )
    base_dict.pop("robots")
    result = OmegaConf.to_container(
        OmegaConf.merge(base_dict, override_dict), resolve=True
    )
    robot["model"] = str(robot.pop("type", robot.get("model", "r1pro"))).lower()
    if robot["model"] != "r1pro":
        raise ValueError(
            "The RLinf legacy256 action/state contract requires model r1pro."
        )
    robot.setdefault("name", "robot_r1")
    # Correct the obsolete spelling in OmniGibson's example config. An explicit
    # modern override always wins.
    old_self_collision = robot.pop("self_collision", None)
    if "self_collisions" not in robot and old_self_collision is not None:
        robot["self_collisions"] = old_self_collision
    proprio = robots[0].get("proprio_obs")
    if proprio is None:
        proprio = R1PRO_PROPRIO_KEYS
    if list(proprio) != R1PRO_PROPRIO_KEYS:
        from omnigibson.eval.utils.eval_utils import PROPRIOCEPTION_INDICES

        if list(proprio) != list(PROPRIOCEPTION_INDICES["R1Pro"]):
            raise ValueError(
                "Use the RLinf or the official 2026 R1Pro proprio_obs field order."
            )
    robot["proprio_obs"] = list(proprio)
    robot["obs_modalities"] = list(
        dict.fromkeys([*robot.get("obs_modalities", []), "proprio", "rgb"])
    )
    result["robots"] = [robot]

    env = result.setdefault("env", {})
    wrapper = env.pop("env_wrapper", "rgb")
    if wrapper is not None and wrapper not in SUPPORTED_ENV_WRAPPERS:
        raise ValueError(f"Unsupported BEHAVIOR observation wrapper: {wrapper!r}")
    if env.get("automatic_reset", False):
        raise ValueError(
            "Set omni_config.env.automatic_reset=false; RLinf owns reset and final observations."
        )
    if env.get("flatten_obs_space", False):
        raise ValueError("RLinf BEHAVIOR requires flatten_obs_space=false.")
    env["num_envs"] = int(num_envs)

    # Sensor resolution must be final before native batched cameras are created.
    camera = result.pop("camera", {}) or {}
    sensor_config = robot.setdefault("sensor_config", {})
    for role, link in CAMERA_LINKS.items():
        resolution = camera.get(
            "head_resolution" if role == "head" else "wrist_resolution"
        )
        if wrapper == "rgb_lowres":
            resolution = (224, 224)
        elif wrapper is not None and resolution is None:
            resolution = RGB_RESOLUTIONS[role]
        if resolution is not None:
            if len(resolution) != 2 or any(int(x) <= 0 for x in resolution):
                raise ValueError(f"Invalid {role} resolution: {resolution!r}")
            sensor = sensor_config.setdefault(link, {}).setdefault("sensor_kwargs", {})
            sensor.update(
                image_height=int(resolution[0]), image_width=int(resolution[1])
            )
    # These are the OLD RLinf wrapper names. In particular, old "default"
    # provides depth/segmentation and is not the new evaluator's RGB-only
    # DefaultWrapper. Declare additions before sensor construction.
    additions = {
        "default": ("depth_linear", "seg_semantic", "seg_instance_id"),
        "rich_obs": ("normal", "flow"),
    }.get(wrapper, ())
    robot["obs_modalities"] = list(
        dict.fromkeys([*robot["obs_modalities"], *additions])
    )

    task = result["task"]
    # RLinf-only fields are consumed by ActivityInstanceLoader, never forwarded to the
    # changed BehaviorTask constructor.
    for key in INSTANCE_KEYS:
        task.pop(key, None)
    task.pop("predefined_problem", None)  # Removed by the compiled BDDL task API.
    task.pop("debug_object_sampling", None)
    task.setdefault("include_obs", False)
    if wrapper == "rich_obs":
        task["include_obs"] = True
    max_steps = OmegaConf.select(cfg, "max_episode_steps")
    if max_steps is None or int(max_steps) < 1:
        raise ValueError("max_episode_steps must be positive.")
    # OmniGibson checks episode_steps before incrementing it in step().
    task.setdefault("termination_config", {})["max_steps"] = int(max_steps) - 1
    scene = result["scene"]
    if scene.pop("partial_scene_load", False):
        from omnigibson.eval.utils.eval_utils import TASK_NAMES_TO_ROOMS

        try:
            scene["load_room_instances"] = deepcopy(
                TASK_NAMES_TO_ROOMS[task["activity_name"]]
            )
        except KeyError as exc:
            raise ValueError(
                f"No 2026 room metadata for task {task['activity_name']!r}."
            ) from exc
        scene["load_room_types"] = None
    result.pop("macro", None)
    return result


def configure_macros(cfg: DictConfig) -> None:
    """Apply supported legacy macros before Environment launches the simulator."""
    from omnigibson.macros import gm

    macro = OmegaConf.select(cfg, "omni_config.macro", default={})
    mapping = {
        "headless": "HEADLESS",
        "enable_object_states": "ENABLE_OBJECT_STATES",
        "use_gpu_dynamics": "USE_GPU_DYNAMICS",
        "enable_transition_rules": "ENABLE_TRANSITION_RULES",
        "render_viewer_camera": "RENDER_VIEWER_CAMERA",
        "use_numpy_controller_backend": "USE_NUMPY_CONTROLLER_BACKEND",
    }
    for key, name in mapping.items():
        if key in macro:
            setattr(gm, name, bool(macro[key]))
    # enable_flatcache has no equivalent in 3.9.3; the simulator now owns its
    # state cache. Do not create a fake macro with no runtime effect.


def validate_simulator_version(cfg: DictConfig) -> None:
    """Reject a config that explicitly requests a different OmniGibson release."""
    version = cfg.get("simulator_version")
    if version not in (None, "3.9.3", "3.9.3-post2"):
        raise ValueError(
            f"Unsupported BEHAVIOR simulator_version {version!r}; expected 3.9.3."
        )


def setup_omni_cfg(cfg: DictConfig) -> DictConfig:
    """Build a single-scene SDK configuration for instance generation."""
    import omnigibson as og

    validate_simulator_version(cfg)
    configure_macros(cfg)
    return OmegaConf.create(
        make_environment_config(cfg, 1, str(Path(og.__file__).parent))
    )
