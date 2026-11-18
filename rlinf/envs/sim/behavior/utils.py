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

import os

import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from rlinf.utils.logging import get_logger

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


CAMERA_LINKS = {
    "head": "zed_link:Camera:0",
    "left_wrist": "left_realsense_link:Camera:0",
    "right_wrist": "right_realsense_link:Camera:0",
}


def set_camera_resolution(robot: DictConfig, camera_cfg, wrapper_name) -> None:
    """Set per-sensor resolution before OmniGibson creates its batched cameras."""
    for role, link in CAMERA_LINKS.items():
        key = "head_resolution" if role == "head" else "wrist_resolution"
        resolution = camera_cfg.get(key)
        if wrapper_name == "rgb_lowres":
            resolution = (224, 224)
        elif resolution is None and wrapper_name is not None:
            resolution = (720, 720) if role == "head" else (480, 480)
        if resolution is not None:
            if len(resolution) != 2 or min(resolution) <= 0:
                raise ValueError(f"Invalid {role} resolution: {resolution!r}")
            OmegaConf.update(
                robot,
                f"sensor_config.{link}.sensor_kwargs",
                {"image_height": resolution[0], "image_width": resolution[1]},
            )


def apply_runtime_renderer_settings() -> None:
    """
    RLinf-specific renderer overrides after OmniGibson has launched.
    """

    import omnigibson.lazy as lazy

    lazy.carb.settings.get_settings().set_float(
        "/rtx-transient/resourcemanager/texturestreaming/memoryBudget",
        0.1,
    )


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


def apply_env_wrapper(env, wrapper_name: str | None):
    if wrapper_name is None:
        return env
    from .rgb_wrapper import RGBWrapper

    # All profiles use the same RGB optics. setup_omni_cfg sets their other
    # modalities and camera dimensions before constructing the batched env.
    return RGBWrapper(env)


def override_sub_cfg(omni_cfg: DictConfig, override_cfg: DictConfig, sub_attr: str):
    omni_sub_cfg = OmegaConf.select(omni_cfg, sub_attr)
    override_sub_cfg = OmegaConf.select(override_cfg, sub_attr)
    if override_sub_cfg is not None:
        setattr(
            omni_cfg,
            sub_attr,
            override_sub_cfg
            if omni_sub_cfg is None
            else OmegaConf.merge(omni_sub_cfg, override_sub_cfg),
        )


def setup_omni_cfg(cfg: DictConfig, num_envs: int = 1) -> DictConfig:
    """
    Setup OmniGibson's config, overrided by user-set config

    Args:
        cfg(DictConfig): rlinf's env config, must have `omni_config` field

    Returns:
        (DictConfig): overrided OmniGibson config
    """
    base_config_name = OmegaConf.select(
        cfg, "base_config_name", default="r1pro_behavior"
    )

    import omnigibson as og
    from omnigibson.macros import gm

    override_cfg = OmegaConf.select(cfg, "omni_config")
    cfg_path = os.path.join(og.example_config_path, f"{base_config_name}.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        omni_cfg = OmegaConf.create(yaml.load(f, Loader=yaml.FullLoader))
    # override env/render/camera/robots/task/scene config
    override_sub_cfg(omni_cfg, override_cfg, "env")
    override_sub_cfg(omni_cfg, override_cfg, "render")
    override_sub_cfg(omni_cfg, override_cfg, "camera")
    override_sub_cfg(omni_cfg, override_cfg, "macro")
    override_sub_cfg(omni_cfg, override_cfg, "task")
    override_sub_cfg(omni_cfg, override_cfg, "scene")
    for key in override_cfg:
        if key not in ("env", "render", "camera", "macro", "task", "scene", "robots"):
            override_sub_cfg(omni_cfg, override_cfg, key)
    # here actually we only needs one robot config (and Behavior actually does do that)
    # we must use update rather than merge to keep default robot config fields.
    robot_override = OmegaConf.select(override_cfg, "robots[0]", default=None)
    assert robot_override is not None and len(override_cfg.robots) == 1, (
        "BEHAVIOR requires exactly one omni_config.robots entry."
    )
    OmegaConf.update(omni_cfg, "robots[0]", robot_override, merge=True)

    override_proprio_obs = OmegaConf.select(
        override_cfg, "robots[0].proprio_obs", default=None
    )
    if override_proprio_obs is None:
        override_proprio_obs = R1PRO_PROPRIO_KEYS
    OmegaConf.update(
        omni_cfg, "robots[0].proprio_obs", override_proprio_obs, merge=True
    )

    # The 3.9.3 constructor identifies the robot by model and the plural
    # self_collisions. The SDK example YAML still uses the old spellings.
    robot = omni_cfg.robots[0]
    robot.model = str(robot.pop("type", robot.get("model", "r1pro"))).lower()
    if robot.model != "r1pro":
        raise ValueError("RLinf BEHAVIOR requires the R1Pro robot.")
    robot.setdefault("name", "robot_r1")
    if "self_collision" in robot:
        robot.setdefault("self_collisions", robot.pop("self_collision"))
    if list(override_proprio_obs) != R1PRO_PROPRIO_KEYS:
        from omnigibson.eval.utils.eval_utils import PROPRIOCEPTION_INDICES

        if list(override_proprio_obs) != list(PROPRIOCEPTION_INDICES["R1Pro"]):
            raise ValueError("Use the RLinf or official 2026 R1Pro proprio_obs order.")

    wrapper = omni_cfg.env.pop("env_wrapper", "rgb")
    if wrapper is not None and wrapper not in SUPPORTED_ENV_WRAPPERS:
        raise ValueError(
            f"Unsupported wrapper name: {wrapper}, expected one of {SUPPORTED_ENV_WRAPPERS}"
        )
    if omni_cfg.env.get("automatic_reset") or omni_cfg.env.get("flatten_obs_space"):
        raise ValueError(
            "RLinf BEHAVIOR requires automatic_reset=false and flatten_obs_space=false."
        )
    omni_cfg.env.num_envs = num_envs
    extra_modalities = {
        "default": ("depth_linear", "seg_semantic", "seg_instance_id"),
        "rich_obs": ("normal", "flow"),
    }.get(wrapper, ())
    robot.obs_modalities = list(
        dict.fromkeys(
            [*robot.get("obs_modalities", []), "proprio", "rgb", *extra_modalities]
        )
    )
    if wrapper == "rich_obs":
        omni_cfg.task.include_obs = True
    else:
        omni_cfg.task.setdefault("include_obs", False)

    # Instance selection belongs to RLinf; the 3.9.3 task constructor also
    # dropped predefined_problem and debug_object_sampling.
    for key in (
        "activity_instance_dir",
        "instance_resample_mode",
        "instance_file_format",
        "instance_split",
        "predefined_problem",
        "debug_object_sampling",
    ):
        omni_cfg.task.pop(key, None)

    # 3.9.3 publishes task-specific room instances in the challenge metadata.
    if omni_cfg.scene.pop("partial_scene_load", False):
        from omnigibson.eval.utils.eval_utils import TASK_NAMES_TO_ROOMS

        activity_name = omni_cfg.task.activity_name
        if activity_name not in TASK_NAMES_TO_ROOMS:
            raise ValueError(f"No 2026 room metadata for task {activity_name!r}.")
        omni_cfg.scene.load_room_instances = TASK_NAMES_TO_ROOMS[activity_name]
        omni_cfg.scene.load_room_types = None
        get_logger().info(
            f"Loaded task-relevant rooms for {activity_name}: {omni_cfg.scene.load_room_instances}"
        )

    # setup omnigibson macros, according to configuration yaml
    for key, value in omni_cfg.pop("macro", {}).items():
        # The old flatcache macro was removed in 3.9.3.
        if key in (
            "headless",
            "enable_object_states",
            "use_gpu_dynamics",
            "enable_transition_rules",
            "render_viewer_camera",
            "use_numpy_controller_backend",
        ):
            setattr(gm, key.upper(), value)

    # setup head/wrist camera resolutions
    set_camera_resolution(robot, omni_cfg.pop("camera", {}) or {}, wrapper)

    # override behavior's termination config `max_steps` field
    max_episode_steps = OmegaConf.select(cfg, "max_episode_steps")
    if max_episode_steps is None or max_episode_steps < 1:
        raise ValueError("max_episode_steps must be positive.")
    OmegaConf.update(
        omni_cfg,
        "task.termination_config.max_steps",
        max_episode_steps - 1,  # OmniGibson checks the counter before incrementing it.
    )

    return omni_cfg


def validate_simulator_version(cfg: DictConfig) -> None:
    """Reject a config that explicitly requests a different OmniGibson release."""
    version = cfg.get("simulator_version")
    if version not in (None, "3.9.3", "3.9.3-post2"):
        raise ValueError(
            f"Unsupported BEHAVIOR simulator_version {version!r}; expected 3.9.3."
        )
