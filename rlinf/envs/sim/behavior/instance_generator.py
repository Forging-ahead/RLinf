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

import argparse
import json
from functools import partial
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

from omegaconf import DictConfig, OmegaConf

from rlinf.envs.sim.behavior.utils import setup_omni_cfg

REPO_ROOT = Path(__file__).resolve().parents[4]


DEFAULT_NUM_TRIALS = 50
DEFAULT_SETTLE_STEPS = 10
DEFAULT_ROBOT_STAGING_POSITION = (-50.0, -50.0, -50.0)
SUPPORTED_OUTPUT_FORMATS = ("template", "tro_state")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for BEHAVIOR cached-instance generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate BEHAVIOR cached activity instances from an RLinf "
            "behavior_r1pro-style yaml."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "examples/embodiment/config/env/behavior_r1pro.yaml"),
        help="Path to an RLinf BEHAVIOR env yaml with an omni_config section.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        required=True,
        choices=SUPPORTED_OUTPUT_FORMATS,
        help="Cached instance format to generate.",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        required=True,
        help="First activity_instance_id to generate.",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        required=True,
        help="Last activity_instance_id to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Initial activity_instance_id seed used to bootstrap the sampler.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help="Maximum sampling trials per activity_instance_id.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to write cached instances into. Defaults to "
            "omni_config.task.activity_instance_dir when set, otherwise RLinf's "
            "default challenge-task-instances directory."
        ),
    )
    parser.add_argument(
        "--robot-staging-position",
        type=float,
        nargs=3,
        default=DEFAULT_ROBOT_STAGING_POSITION,
        metavar=("X", "Y", "Z"),
        help="Temporary robot position used during online object sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cached instance files.",
    )
    return parser.parse_args()


def load_env_cfg(config_path: str) -> DictConfig:
    """Load an RLinf BEHAVIOR env yaml.

    Args:
        config_path: Path to the yaml file.

    Returns:
        The loaded OmegaConf config.

    Raises:
        ValueError: If the yaml does not contain an ``omni_config`` section.
    """
    env_cfg = OmegaConf.load(config_path)
    if OmegaConf.select(env_cfg, "omni_config") is None:
        raise ValueError(
            f"{config_path} must be an RLinf BEHAVIOR env yaml with an omni_config section."
        )
    return env_cfg


def build_sampling_omni_cfg(
    env_cfg: DictConfig,
    seed: int,
    robot_staging_position: tuple[float, float, float],
) -> DictConfig:
    """Build the OmniGibson config used for cached-instance sampling.

    Args:
        env_cfg: RLinf BEHAVIOR env yaml.
        seed: Initial ``activity_instance_id`` used to bootstrap sampling.
        robot_staging_position: Temporary robot position during sampling.

    Returns:
        The OmniGibson config used to create the sampling environment.
    """
    from omnigibson.tasks.behavior_task import BehaviorTask
    from omnigibson.utils.asset_utils import get_task_instance_path

    omni_cfg = setup_omni_cfg(env_cfg)
    task = omni_cfg.task
    # Like the SDK's multiply_b1k_tasks.py, resample from the canonical
    # template with its compiled BDDL object scope already bound.
    scene = omni_cfg.scene.scene_model
    template = BehaviorTask.get_cached_activity_scene_filename(
        scene, task.activity_name, task.get("activity_definition_id", 0), 0
    )
    path = get_task_instance_path(scene, template, mode="train")
    if path is None:
        raise FileNotFoundError(f"Missing canonical 2026 task template: {template}")
    OmegaConf.update(omni_cfg, "env.automatic_reset", False)
    OmegaConf.update(omni_cfg, "task.activity_instance_id", seed)
    OmegaConf.update(omni_cfg, "task.online_object_sampling", False)
    OmegaConf.update(omni_cfg, "task.use_presampled_robot_pose", False)
    OmegaConf.update(omni_cfg, "scene.scene_instance", None, merge=False)
    OmegaConf.update(omni_cfg, "scene.scene_file", path, merge=False)
    for idx, _ in enumerate(OmegaConf.select(omni_cfg, "robots", default=[])):
        OmegaConf.update(
            omni_cfg,
            f"robots[{idx}].position",
            list(robot_staging_position),
            merge=False,
        )
    return omni_cfg


def resolve_output_dir(omni_cfg: DictConfig, explicit_output_dir: str | None) -> Path:
    """Resolve the directory used to save cached instances.

    Args:
        omni_cfg: OmniGibson config used for sampling.
        explicit_output_dir: CLI override directory.

    Returns:
        The directory where cached instances should be written.
    """
    if explicit_output_dir is not None:
        return Path(explicit_output_dir)

    activity_instance_dir = OmegaConf.select(omni_cfg, "task.activity_instance_dir")
    if activity_instance_dir is not None:
        return Path(activity_instance_dir)

    from omnigibson.utils.asset_utils import get_dataset_path

    scene_model = OmegaConf.select(omni_cfg, "scene.scene_model")
    activity_name = OmegaConf.select(omni_cfg, "task.activity_name")
    return (
        Path(get_dataset_path("2026-challenge-task-instances"))
        / "scenes"
        / scene_model
        / "json"
        / f"{scene_model}_task_{activity_name}_instances"
    )


def configure_sampling_macros() -> None:
    """Apply the OmniGibson sampling macro settings used by BEHAVIOR scripts."""
    from omnigibson.macros import macros

    macros.systems.micro_particle_system.MICRO_PARTICLE_SYSTEM_MAX_VELOCITY = 0.5
    macros.systems.macro_particle_system.MACRO_PARTICLE_SYSTEM_MAX_DENSITY = 200.0
    macros.utils.object_state_utils.DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS = 5
    macros.utils.object_state_utils.DEFAULT_LOW_LEVEL_SAMPLING_ATTEMPTS = 5


def build_output_path(task, output_dir: Path, output_format: str) -> Path:
    """Build the output file path for the current task instance.

    Args:
        task: Active OmniGibson BEHAVIOR task.
        output_dir: Directory that stores cached instances.
        output_format: Cached instance format.

    Returns:
        The full output path for the current ``activity_instance_id``.

    Raises:
        ValueError: If ``output_format`` is unsupported.
    """
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(
            f"Unsupported output_format {output_format!r}. Expected one of {SUPPORTED_OUTPUT_FORMATS}."
        )

    filename = task.get_cached_activity_scene_filename(
        scene_model=task.scene_name,
        activity_name=task.activity_name,
        activity_definition_id=task.activity_definition_id,
        activity_instance_id=task.activity_instance_id,
    )
    suffix = ".json" if output_format == "template" else "-tro_state.json"
    return output_dir / f"{filename}{suffix}"


def dump_tro_state(
    env,
    output_path: Path,
    overwrite: bool,
    capture_current_robot_pose: bool = False,
) -> None:
    """Write the current task-relevant state to ``output_path``.

    Args:
        env: Active OmniGibson environment.
        output_path: Target ``*_template-tro_state.json`` file.
        overwrite: Whether existing files may be overwritten.
        capture_current_robot_pose: When True, record the robot's *current*
            base pose (so a subsequent ``load_activity_instance_tro_state``
            restores where the robot is right now). When False (default),
            preserve the presampled ``robot_poses`` metadata used to bootstrap
            future resets. Offline sampling (``instance_generator.main``)
            wants False; mid-rollout snapshots want True.

    Raises:
        FileExistsError: If ``output_path`` exists and ``overwrite`` is false.
    """
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing cached instance: {output_path}"
        )

    from omnigibson.utils.config_utils import TorchEncoder

    # ``dump_state`` returns root_link poses in world frame, but the loader
    # interprets them as scene-relative when ``env.scene.idx != 0``. Rebase to
    # scene frame here so dump/load are symmetric across scene indices.
    scene = env.scenes[0]
    rebase_to_scene = scene.idx != 0
    tro_state = {}
    for bddl_name, bddl_inst in env.task.object_scopes[0].items():
        if bddl_inst is None or bddl_name.startswith("agent."):
            continue
        state = bddl_inst.dump_state(serialized=False)
        if (
            rebase_to_scene
            and isinstance(state, dict)
            and isinstance(state.get("root_link"), dict)
            and "pos" in state["root_link"]
            and "ori" in state["root_link"]
        ):
            rebased_root_link = dict(state["root_link"])
            rebased_pos, rebased_ori = scene.convert_world_pose_to_scene_relative(
                rebased_root_link["pos"],
                rebased_root_link["ori"],
            )
            rebased_root_link["pos"] = rebased_pos
            rebased_root_link["ori"] = rebased_ori
            state = dict(state)
            state["root_link"] = rebased_root_link
        tro_state[bddl_name] = state
    robot_poses = scene.get_task_metadata(key="robot_poses")
    if capture_current_robot_pose:
        # Mid-rollout dump: overwrite the presampled pose entry with the
        # robot's actual current scene-frame base pose, so a subsequent load
        # restores where the robot is right now (otherwise the loader would
        # teleport it back to the reset-time presampled pose).
        robot = env.task.get_agent(env, env_idx=0)
        cur_pos, cur_ori = robot.get_position_orientation(frame="scene")
        # The SDK selects the generic "robot" entry before a model-specific one.
        robot_poses = dict(robot_poses or {})
        robot_poses["robot"] = [{"position": cur_pos, "orientation": cur_ori}]
    if robot_poses is not None:
        tro_state["robot_poses"] = robot_poses

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(tro_state, f, cls=TorchEncoder, indent=4)


def save_activity_instance(
    env,
    output_dir: Path,
    output_format: str,
    overwrite: bool,
) -> Path:
    """Save the current sampled instance in the requested format.

    Args:
        env: Active OmniGibson environment.
        output_dir: Directory that stores cached instances.
        output_format: Cached instance format.
        overwrite: Whether existing files may be overwritten.

    Returns:
        The written output path.

    Raises:
        FileExistsError: If the target file exists and ``overwrite`` is false.
    """
    output_path = build_output_path(env.task, output_dir, output_format)
    if output_format == "template":
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing cached instance: {output_path}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        env.task.save_task(
            env=env,
            save_dir=str(output_dir),
            override=True,
            task_relevant_only=False,
        )
    else:
        dump_tro_state(env, output_path=output_path, overwrite=overwrite)
    return output_path


def _validate_sample(env, scene_state, stable_scene, validate_task) -> None:
    """Validate the sampled scene with the SDK's single-scene sampling script."""
    from omnigibson.macros import gm

    # The sampling validator accepts unbatched predicates. Bind its one scene.
    validation_task = SimpleNamespace(
        _base_conditions=env.task._base_conditions,
        activity_initial_conditions=env.task.activity_initial_conditions,
        _evaluate_predicate=partial(env.task._evaluate_predicate, 0),
    )
    transition_rules = gm.ENABLE_TRANSITION_RULES
    try:
        validate_task(
            validation_task,
            {"state": scene_state},
            stable_scene,
            active_room_instances=env.scenes[0].load_room_instances,
        )
    finally:
        # The SDK restores this flag only when validation succeeds.
        with gm.unlocked():
            gm.ENABLE_TRANSITION_RULES = transition_rules


def generate_activity_instances(
    env,
    output_dir: Path,
    output_format: str,
    start_idx: int,
    end_idx: int,
    num_trials: int,
    overwrite: bool,
) -> None:
    """Generate cached BEHAVIOR instances by repeated online sampling.

    Args:
        env: Active OmniGibson environment configured for online object sampling.
        output_dir: Directory that stores cached instances.
        output_format: Cached instance format.
        start_idx: First activity instance id to generate.
        end_idx: Last activity instance id to generate.
        num_trials: Maximum trials per instance id.
        overwrite: Whether existing files may be overwritten.

    Raises:
        RuntimeError: If any requested instance id could not be generated.
        ValueError: If ``start_idx`` / ``end_idx`` are invalid.
    """
    if start_idx > end_idx:
        raise ValueError(
            f"start_idx must be <= end_idx, got {start_idx} and {end_idx}."
        )
    if num_trials < 1:
        raise ValueError(f"num_trials must be >= 1, got {num_trials}.")

    import omnigibson as og
    from omnigibson.objects import DatasetObject
    from omnigibson.utils.asset_utils import get_dataset_path

    # The official validator ships in the SDK source checkout, alongside the
    # sampling CLI. Load it from the same checkout as the running simulator.
    sampling_utils = (
        Path(og.__file__).resolve().parents[1] / "scripts/sampling/utils.py"
    )
    validate_task = run_path(str(sampling_utils))["validate_task"]
    scene = env.scenes[0]
    default_scene_path = (
        Path(get_dataset_path("2026-challenge-task-instances"))
        / "scenes"
        / env.task.scene_name
        / "json"
        / f"{env.task.scene_name}_stable.json"
    )
    if not default_scene_path.is_file():
        raise ValueError(f"Stable scene json does not exist: {default_scene_path}")

    with default_scene_path.open("r", encoding="utf-8") as f:
        default_scene_dict = json.load(f)

    env.task.sampler._compiled_task = env.task.compiled_task
    env.task.sampler._parse_inroom_object_room_assignment()
    env.task.sampler._build_sampling_order()

    for system in scene.active_systems.values():
        system.remove_all_particles()
    og.sim.step()
    initial_state = og.sim.dump_state()

    failed_instance_ids = []
    for instance_id in range(start_idx, end_idx + 1):
        env.task.activity_instance_id = instance_id
        output_path = build_output_path(env.task, output_dir, output_format)
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing cached instance: {output_path}"
            )

        print(
            f"Sampling activity_instance_id={instance_id} -> {output_path.name}",
            flush=True,
        )
        for trial_idx in range(num_trials):
            og.sim.load_state(initial_state)
            og.sim.step()

            error_msg = env.task.sampler._sample_initial_conditions_final()
            if error_msg is not None:
                print(
                    f"  trial {trial_idx}: sampling failed: {error_msg}",
                    flush=True,
                )
                continue

            for _ in range(DEFAULT_SETTLE_STEPS):
                og.sim.step()

            for entity in env.task.object_scopes[0].values():
                if isinstance(entity, DatasetObject):
                    entity.keep_still()

            for _ in range(DEFAULT_SETTLE_STEPS):
                og.sim.step()

            task_final_state = scene.dump_state()
            try:
                _validate_sample(
                    env, task_final_state, default_scene_dict, validate_task
                )
            except ValueError as error:
                print(f"  trial {trial_idx}: validation failed: {error}", flush=True)
                continue

            scene.load_state(task_final_state)
            scene.update_initial_file()
            saved_path = save_activity_instance(
                env,
                output_dir=output_dir,
                output_format=output_format,
                overwrite=overwrite,
            )
            print(f"  trial {trial_idx}: saved {saved_path}", flush=True)
            break
        else:
            failed_instance_ids.append(instance_id)

    if failed_instance_ids:
        raise RuntimeError(
            "Failed to generate cached instances for "
            f"activity_instance_id(s): {failed_instance_ids}"
        )


def main() -> None:
    """Run BEHAVIOR cached-instance generation."""
    args = parse_args()
    env_cfg = load_env_cfg(args.config)
    configured_output_dir = OmegaConf.select(
        env_cfg,
        "omni_config.task.activity_instance_dir",
    )

    print("Building OmniGibson sampling config...", flush=True)
    omni_cfg = build_sampling_omni_cfg(
        env_cfg=env_cfg,
        seed=args.seed,
        robot_staging_position=tuple(args.robot_staging_position),
    )
    output_dir = resolve_output_dir(
        omni_cfg,
        args.output_dir if args.output_dir is not None else configured_output_dir,
    )
    print(f"Cached instances will be written to: {output_dir}", flush=True)

    configure_sampling_macros()

    import omnigibson as og

    env = None
    try:
        omni_cfg_dict = OmegaConf.to_container(
            omni_cfg,
            resolve=True,
            throw_on_missing=True,
        )
        env = og.Environment(omni_cfg_dict)
        generate_activity_instances(
            env,
            output_dir=output_dir,
            output_format=args.output_format,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            num_trials=args.num_trials,
            overwrite=args.overwrite,
        )
    finally:
        if env is not None:
            env.close()
    # Kit shutdown can terminate Python. Let sampling exceptions propagate.
    og.shutdown()


if __name__ == "__main__":
    main()
