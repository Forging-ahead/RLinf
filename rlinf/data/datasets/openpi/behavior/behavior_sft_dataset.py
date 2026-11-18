# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import bisect
import dataclasses
import json
import logging
import os
import random
from collections import defaultdict
from collections.abc import Callable, Iterable
from functools import cached_property
from itertools import accumulate
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch as th
import torch.distributed as dist
from datasets import Dataset
from huggingface_hub import snapshot_download
from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import (
    CODEBASE_VERSION,
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.datasets.utils import (
    check_delta_timestamps,
    get_delta_indices,
    get_hf_features_from_features,
    hf_transform_to_torch,
    load_json,
    load_stats,
    unflatten_dict,
)
from lerobot.datasets.video_utils import get_safe_default_codec
from torch.utils.data import get_worker_info

from rlinf.data.datasets.openpi.transform_fn import DataTransformFn

logger = logging.getLogger("BehaviorSftDataset")

# The v3 release packs multiple episodes into each camera file.
ROBOT_CAMERA_NAMES = {
    "R1Pro": {
        "left_wrist": "left_realsense_link_camera_0",
        "right_wrist": "right_realsense_link_camera_0",
        "head": "zed_link_camera_0",
    },
}
ORCHESTRATORS_PATH = "orchestrators"


def _stream_video_frames(path: Path, start: float):
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        container.seek(int(start / video.time_base), stream=video)
        for frame in container.decode(video):
            if frame.pts is not None:
                yield frame


def _depth_in_meters(frame, info):
    # 2026 encodes depth in 12 bits. Keep RLinf's original single-channel output in meters.
    norm = frame.to_ndarray(format="gray12le").astype(np.float32) / 4095
    near, far = info["video.depth_min"], info["video.depth_max"]
    if info["video.use_log"]:
        shift = info["video.shift"]
        lo, hi = np.log(near + shift), np.log(far + shift)
        depth = np.exp(norm * (hi - lo) + lo) - shift
    else:
        depth = near + norm * (far - near)
    return th.from_numpy(np.clip(depth, near, far)).float().unsqueeze(0)


# ---------------------------------------------------------------------------
# PromptFromLeRobotItem transform
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotItem(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset item's ``task`` field."""

    def __call__(self, data: dict) -> dict:
        return {**data, "prompt": data.pop("task")}


# ---------------------------------------------------------------------------
# Orchestrator helpers
# ---------------------------------------------------------------------------


def load_orchestrators_data(episode_path_or_level_0_task, episode_len):
    """Build the per-level (task, start_frame, end_frame) orchestrator table."""
    output_data = defaultdict(list)
    if type(episode_path_or_level_0_task) is str:
        for i in range(4):
            output_data[i] = [
                {
                    "task": episode_path_or_level_0_task,
                    "start_frame": 0,
                    "end_frame": episode_len - 1,
                }
            ]
        return output_data
    episode_path = episode_path_or_level_0_task
    task_annotated_data = load_json(episode_path / "task_annotated.json")
    level_0_task = task_annotated_data["cot_task_description"]
    output_data[0].append(
        {
            "task": level_0_task,
            "start_frame": 0,
            "end_frame": episode_len - 1,
        }
    )
    try:
        num_level1_tasks = len(task_annotated_data["cot_subtask_description_list"])
        for i in range(num_level1_tasks):
            subtask_data = load_json(episode_path / f"subtask_{i}_annotated.json")
            subtask = subtask_data["cot_subtask_description"]
            start_frame, end_frame = (
                subtask_data["start_frame"],
                subtask_data["end_frame"] - 1,
            )
            skill = subtask_data["skill_description"]
            output_data[1].append(
                {"task": skill, "start_frame": start_frame, "end_frame": end_frame}
            )
            output_data[2].append(
                {"task": subtask, "start_frame": start_frame, "end_frame": end_frame}
            )
            for event_data_path in sorted(
                episode_path.glob(f"event_{i}_*_annotated.json")
            ):
                event_data = load_json(event_data_path)
                event_task = event_data["subtask_answer_detailed"]
                start_frame, end_frame = (
                    event_data["start_frame"],
                    event_data["end_frame"] - 1,
                )
                output_data[3].append(
                    {
                        "task": event_task,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                    }
                )
    except Exception as e:
        logger.warning(
            "%s failed to load orchestrators data: %s, falling back to default task.",
            episode_path,
            e,
        )
        for i in range(len(output_data)):
            output_data[i] = output_data[0]
    return output_data


def skill_weight(cur_skill, skill_list: list[str]) -> float:
    """Return the sampling weight for ``cur_skill`` given a ``skill:weight`` list."""
    if "all" in skill_list:
        skill_list = [skill for skill in skill_list if skill != "all"]
        for skill_item in skill_list:
            skill, weight = skill_item.split(":")
            if skill == cur_skill:
                return float(weight)
        return 1.0
    for skill_item in skill_list:
        skill, weight = skill_item.split(":")
        if skill == cur_skill:
            return float(weight)
    return 0.0


# ---------------------------------------------------------------------------
# BehaviorSftDatasetMetadata
# ---------------------------------------------------------------------------


class BehaviorSftDatasetMetadata(LeRobotDatasetMetadata):
    """LeRobot v3 metadata extended with BEHAVIOR task and camera selection."""

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
        tasks: Iterable[str] | None = None,
        modalities: Iterable[str] | None = None,
        cameras: Iterable[str] | None = None,
        local_only: bool = False,
    ):
        self.task_name_candidates = None if tasks is None else set(tasks)
        self.modalities = {
            "depth_linear" if mod == "depth" else mod for mod in modalities or ["rgb"]
        }
        camera_names = ROBOT_CAMERA_NAMES["R1Pro"]
        self.camera_names = {
            camera_names.get(name, name) for name in cameras or camera_names
        }
        if not self.modalities <= {"rgb", "depth_linear"}:
            raise ValueError(
                f"Unsupported BEHAVIOR 2026 video modalities: {sorted(self.modalities - {'rgb', 'depth_linear'})}. "
                "The demonstrations publish RGB and depth_linear videos."
            )
        self._selected_video_keys = {
            f"observation.{mod}.{cam}"
            for mod in self.modalities
            for cam in self.camera_names
        }
        if local_only:
            if force_cache_sync:
                raise ValueError("force_cache_sync requires local_only=False.")
            self.repo_id = repo_id
            self.revision = revision or CODEBASE_VERSION
            self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
            self.load_metadata()
        else:
            super().__init__(repo_id, root, revision, force_cache_sync)

    def load_metadata(self):
        """Load standard v3 tables, then select the BEHAVIOR tasks to read."""
        with (self.root / "meta/info.json").open() as file:
            version = json.load(file)["codebase_version"]
        if version != "v3.0":
            raise ValueError(
                f"BEHAVIOR SFT requires 2026 demonstrations (v3.0), found {version!r}."
            )
        super().load_metadata()
        self.__dict__.pop("annotations", None)
        self.__dict__.pop("orchestrators", None)
        descriptions, self.task_to_task_index, names = self.load_tasks(self.root)
        self.task_prompts = {name: descriptions[index] for index, name in names.items()}
        self.task_names = set(names.values())
        if self.task_name_candidates is not None:
            self.task_names &= self.task_name_candidates
        self.selected_episodes = self.load_episodes(self.root)
        self.episodes_stats = self.load_episodes_stats(self.root)
        if self.episodes_stats:
            self.stats = aggregate_stats(list(self.episodes_stats.values()))

        videos = {
            key
            for key, value in self.info["features"].items()
            if value["dtype"] == "video"
        }
        if videos:
            missing = self._selected_video_keys - videos
            if missing:
                raise ValueError(f"Missing BEHAVIOR camera features: {sorted(missing)}")

    def load_tasks(self, local_dir: Path):
        """Read the BEHAVIOR task descriptions that supplement the v3 task table."""
        with (local_dir / "meta/tasks.jsonl").open() as file:
            rows = list(map(json.loads, file))
        names, tasks = {}, {}
        for row in rows:
            name = row["task_name"]
            index = row.get("task_index")
            if index is None:
                index = int(self.tasks.loc[name, "task_index"])
            names[index] = name
            tasks[index] = row["task"]
        return tasks, {text: idx for idx, text in tasks.items()}, names

    def load_episodes(self, local_dir: Path):
        """Select episodes without renumbering the v3 table LeRobot uses for videos."""
        columns = [
            "episode_index",
            "tasks",
            "length",
            "data/chunk_index",
            "data/file_index",
        ]
        for key in self.video_keys:
            columns.extend([f"videos/{key}/chunk_index", f"videos/{key}/file_index"])
        columns.extend(
            key
            for key in ("annotation_path", "metainfo_path")
            if key in self.episodes.column_names
        )
        return {
            row["episode_index"]: row
            for row in self.episodes.select_columns(columns)
            if self.task_name_candidates is None
            or not self.task_names.isdisjoint(row["tasks"])
        }

    def load_stats(self, local_dir: Path):
        """Read the published aggregate statistics from meta/stats.json."""
        return load_stats(local_dir)

    def load_episodes_stats(self, local_dir: Path):
        """Read selected episode statistics when the v3 Parquet includes stats columns."""
        import pyarrow.dataset as ds

        source = ds.dataset(local_dir / "meta/episodes", format="parquet")
        columns = [key for key in source.schema.names if key.startswith("stats/")]
        if not columns or not self.selected_episodes:
            return {}
        rows = source.to_table(
            columns=["episode_index", *columns],
            filter=ds.field("episode_index").isin(list(self.selected_episodes)),
        )
        return {
            row["episode_index"]: unflatten_dict(
                {key.removeprefix("stats/"): np.asarray(row[key]) for key in columns}
            )
            for row in rows.to_pylist()
        }

    def load_annotations(self, local_dir: Path, episodes=None):
        """Load available skill annotations using the v3 episode-to-source mapping."""
        ids = self.selected_episodes if episodes is None else episodes
        annotations = {}
        for ep_id in ids:
            if self.selected_episodes[ep_id].get("annotation_path"):
                path = local_dir / self.get_annotation_path(ep_id)
                if path.is_file():
                    with path.open() as file:
                        annotations[ep_id] = json.load(file)
        return annotations

    def load_orchestrators(self, local_dir: Path, episodes=None):
        """Read optional hierarchies, falling back to the episode's main instruction."""
        ids = self.selected_episodes if episodes is None else episodes
        orchestrators = {}
        for ep_id in ids:
            row = self.selected_episodes[ep_id]
            name = row["tasks"][0]
            source = self.task_prompts.get(name, name)
            if row.get("annotation_path"):
                annotation = self.get_annotation_path(ep_id)
                directory = (
                    local_dir
                    / ORCHESTRATORS_PATH
                    / annotation.parent.name
                    / annotation.stem
                )
                if directory.is_dir():
                    source = directory
            orchestrators[ep_id] = load_orchestrators_data(source, row["length"])
        return orchestrators

    @cached_property
    def annotations(self):
        return self.load_annotations(self.root)

    @cached_property
    def orchestrators(self):
        return self.load_orchestrators(self.root)

    def get_annotation_path(self, ep_index: int) -> Path:
        """Resolve the original annotation filename stored on this v3 episode."""
        return Path(self.selected_episodes[ep_index]["annotation_path"])

    def get_metainfo_path(self, ep_index: int) -> Path:
        """Resolve a supplemental metainfo path when the episode publishes one."""
        return Path(self.selected_episodes[ep_index]["metainfo_path"])

    @property
    def annotation_path(self) -> str | None:
        """Optional dataset-wide path template; 2026 publishes paths per episode."""
        return self.info.get("annotation_path")

    @property
    def metainfo_path(self) -> str | None:
        """Optional dataset-wide supplemental metainfo template."""
        return self.info.get("metainfo_path")

    @property
    def features(self) -> dict[str, dict]:
        """Keep state and action columns along with the selected camera streams."""
        return {
            key: feature
            for key, feature in self.info["features"].items()
            if feature["dtype"] != "video" or key in self._selected_video_keys
        }


# ---------------------------------------------------------------------------
# BehaviorSftDataset
# ---------------------------------------------------------------------------


def partition_chunk_indices(
    num_chunks: int,
    *,
    rank: int,
    world_size: int,
    worker_id: int,
    num_workers: int,
) -> list[int]:
    """Return the chunk indices a single ``(rank, worker)`` pair streams.

    The distributed ``rank`` is folded into the per-worker stride so that every
    ``(rank, worker)`` pair receives a disjoint, non-overlapping set of chunk
    indices whose union over all ranks and workers covers every chunk. This is
    what lets the streaming dataset shard distinct data per rank under
    ``torchrun``/DDP: a ``DistributedSampler`` cannot, because the dataset
    ignores the ``idx`` it is handed and streams chunks internally.
    """
    global_worker_id = rank * num_workers + worker_id
    stride = world_size * num_workers
    return list(range(global_worker_id, num_chunks, stride))


class BehaviorSftDataset(LeRobotDataset):
    """Streaming BEHAVIOR-1K dataset for pi05 SFT (task filtering + chunk streaming).

    The dataset streams contiguous keyframe *chunks* of each episode rather than
    returning the frame at a random ``idx``: ``__getitem__`` ignores ``idx`` and
    advances an internal streaming cursor. Chunks are partitioned across data
    loader workers (and, in this port, across distributed ranks) so that every
    consumer sees a disjoint stream — see :meth:`__getitem__`.

    The task prompt uses the demonstration's main instruction or a configured
    skill label selected from its annotated frame windows.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = "pyav",
        batch_encoding_size: int = 1,
        # Custom arguments
        tasks: Iterable[str] | None = None,
        modalities: Iterable[str] | None = None,
        cameras: Iterable[str] | None = None,
        local_only: bool = False,
        check_timestamp_sync: bool = True,
        chunk_streaming_using_keyframe: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        fine_grained_level: int = 0,
        train_rgb_type: str = "regular",
        return_seg_instance: bool = False,
        skill_list: list[str] | None = None,
        skill_labels: dict[int, str] | None = None,
        use_skill: bool = False,
        enable_gap: bool = True,
        allow_left: int = 0,
        allow_right: int = 0,
        dist_rank: int | None = None,
        dist_world_size: int | None = None,
        action_horizon: int | None = None,
    ):
        if skill_list is None:
            skill_list = ["all"]

        th.utils.data.Dataset.__init__(self)
        self.repo_id = repo_id
        self.root = (
            Path(os.path.expanduser(str(root))) if root else HF_LEROBOT_HOME / repo_id
        )
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s
        self.revision = revision or CODEBASE_VERSION
        self.video_backend = video_backend or get_safe_default_codec()
        self.delta_indices = None
        self.batch_encoding_size = batch_encoding_size
        self.episodes_since_last_encoding = 0
        self.return_seg_instance = return_seg_instance
        self.train_rgb_type = train_rgb_type
        self.skill_list = skill_list
        self.skill_labels = skill_labels
        # When `use_skill` is set, the training prompt is the per-frame SKILL text
        # (resolved by the window logic) instead of the main-task text; explicit
        # `skill_labels` are required in that case (the production builder sources
        # them from config), otherwise the constructor raises below.
        self.use_skill = use_skill
        # Skill-mode windowing, aligned with the JAX openpi-comet semantics:
        # `enable_gap` absorbs a true gap into both adjacent skills (so the gap
        # region overlaps and is shared); `allow_left` / `allow_right` are frame
        # counts that extend contiguous skill boundaries outward (also creating
        # overlap). See `_build_skill_boundaries` / `_get_skill_label`.
        self.enable_gap = enable_gap
        self.allow_left = allow_left
        self.allow_right = allow_right
        # Explicit distributed identity captured in the MAIN process. The streaming
        # chunk partition is rank-aware, but DataLoader workers are SPAWNED (fresh
        # interpreters that do NOT inherit ``torch.distributed``), so reading
        # ``dist.get_rank()`` inside a worker returns 0 and every rank would replicate
        # rank 0's partition. Storing rank/world_size here (pickled into the worker)
        # lets ``_select_streaming_chunk`` partition by the correct per-rank id; we fall
        # back to ``torch.distributed`` only when these are not provided.
        self._dist_rank = dist_rank
        self._dist_world_size = dist_world_size
        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        self.seed = seed
        if modalities is None:
            modalities = ["rgb"]
        if cameras is None:
            cameras = ["head", "left_wrist", "right_wrist"]
        if train_rgb_type != "regular":
            raise ValueError(
                "The 2026 demonstrations publish only regular RGB camera streams."
            )
        if return_seg_instance:
            raise ValueError(
                "The 2026 demonstrations publish RGB and depth, without instance masks."
            )
        self.fine_grained_level = fine_grained_level
        self.task_names = set(tasks) if tasks is not None else set()

        self.meta = BehaviorSftDatasetMetadata(
            repo_id=self.repo_id,
            root=self.root,
            revision=self.revision,
            force_cache_sync=force_cache_sync,
            tasks=self.task_names or None,
            modalities=modalities,
            cameras=cameras,
            local_only=local_only,
        )
        self.revision = self.meta.revision
        self.task_prompts = self.meta.task_prompts
        # v3 stores all episodes of a task in shared Parquet and video files.
        # Keep the published row order while selecting positions within each task.
        rows = pd.DataFrame.from_records(list(self.meta.selected_episodes.values()))
        if episodes is not None and not rows.empty:
            by_task = rows[["episode_index", "tasks"]].explode("tasks")
            if self.task_names:
                # A multi-task episode can occupy a different position in each task.
                by_task = by_task[by_task.tasks.isin(self.task_names)]
            by_task = by_task.groupby("tasks", sort=False).nth(episodes)
            rows = rows[rows.episode_index.isin(by_task.episode_index)]
        if rows.empty:
            raise ValueError(
                f"No BEHAVIOR 2026 episodes match {sorted(self.task_names)}."
            )
        self.task_names = self.meta.task_names
        self.task_indices = sorted(
            int(self.meta.tasks.loc[name, "task_index"]) for name in self.task_names
        )
        self._episode_rows = rows
        self.episodes = rows.episode_index.tolist()
        shards = rows.drop_duplicates(["data/chunk_index", "data/file_index"])
        self._data_file_paths = [
            self.meta.get_data_file_path(i) for i in shards.episode_index
        ]
        self.episode_data_index_pos = {
            ep_idx: i for i, ep_idx in enumerate(self.episodes)
        }
        logger.info("Total episodes: %d", len(self.episodes))
        self.stats = self.meta.stats
        if self.meta.episodes_stats and len(self.episodes) != len(
            self.meta.selected_episodes
        ):
            self.stats = aggregate_stats(
                [self.meta.episodes_stats[i] for i in self.episodes]
            )

        try:
            if force_cache_sync and not local_only:
                raise FileNotFoundError
            for fpath in self.get_episodes_file_paths():
                if not download_videos and fpath.startswith("videos/"):
                    continue
                if not (self.root / fpath).is_file():
                    raise FileNotFoundError(self.root / fpath)
            self.hf_dataset = self.load_hf_dataset()
        except (FileNotFoundError, NotADirectoryError):
            if local_only:
                raise
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        ends = rows.length.cumsum()
        starts = ends - rows.length
        if len(self.hf_dataset) != ends.iloc[-1] or (
            self.hf_dataset.data.column("episode_index")
            .take(starts.tolist())
            .to_pylist()
            != self.episodes
            or self.hf_dataset.data.column("episode_index")
            .take((ends - 1).tolist())
            .to_pylist()
            != self.episodes
        ):
            raise ValueError(
                "Local BEHAVIOR frames do not match the selected episode lengths."
            )
        self.episode_bounds = dict(zip(self.episodes, zip(starts, ends)))
        self.episode_data_index = {
            "from": th.from_numpy(starts.to_numpy(copy=True)),
            "to": th.from_numpy(ends.to_numpy(copy=True)),
        }
        if check_timestamp_sync:
            # Check the published FPS within each episode without materializing the entire dataset.
            timestamps = self.hf_dataset.data.column("timestamp")
            for ep_id, (start, stop) in self.episode_bounds.items():
                times = timestamps.slice(start, stop - start).to_numpy()
                if not np.isfinite(times).all() or not np.allclose(
                    np.diff(times), 1 / self.fps, rtol=0, atol=self.tolerance_s
                ):
                    raise ValueError(
                        f"Timestamps in BEHAVIOR episode {ep_id} do not match {self.fps} FPS."
                    )

        if action_horizon is not None:
            if self.delta_timestamps is not None:
                raise ValueError("Specify either action_horizon or delta_timestamps.")
            self.delta_timestamps = {
                "action": [i / self.fps for i in range(action_horizon)]
            }
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

        self.prepare_task(fine_grained_level)

        if self.use_skill and not self.skill_labels:
            raise ValueError(
                "BehaviorSftDataset(use_skill=True) requires explicit skill_labels."
            )
        self.frame_ranges = list(self.episode_bounds.values())
        if self.skill_labels is not None:
            self._build_skill_boundaries()
        self._range_ends = list(
            accumulate(end - start for start, end in self.frame_ranges)
        )
        self._chunk_streaming_using_keyframe = chunk_streaming_using_keyframe
        self.shuffle = shuffle
        if self._chunk_streaming_using_keyframe:
            self.chunks = self._get_keyframe_chunk_indices()
            self.current_streaming_chunk_idx = None
            self.current_streaming_frame_idx = None
            self.obs_loaders = {}

    def prepare_task(self, fine_grained_level: int):
        """Pre-compute cumulative subtask end-frames per episode for lookups."""
        self.fine_grained_level = fine_grained_level
        self.task_sizes = {}
        if (
            fine_grained_level
            or self.skill_list != ["all"]
            or (self.root / ORCHESTRATORS_PATH).is_dir()
        ):
            self.meta.orchestrators = self.meta.load_orchestrators(
                self.root, self.episodes
            )
            for ep_id, ep_orch in self.meta.orchestrators.items():
                self.task_sizes[ep_id] = [
                    task_info["end_frame"] for task_info in ep_orch[fine_grained_level]
                ]

    def _build_skill_boundaries(self):
        """Build per-episode *effective* skill windows, aligned with openpi-comet.

        For each skill, the effective window extends a contiguous (no-gap)
        boundary outward by ``allow_left`` / ``allow_right`` frames, and — when
        ``enable_gap`` is set — absorbs a true gap into both adjacent skills so
        the gap region overlaps and is shared. Windows are clamped to the
        episode's valid duration. Frames inside more than one window are resolved
        per-sample by a random choice in :meth:`_get_skill_label`.
        """
        self.skill_start_frames: dict[int, list[int]] = {}
        self.skill_end_frames: dict[int, list[int]] = {}
        self.meta.annotations = self.meta.load_annotations(self.root, self.episodes)
        eligible = []
        for ep_id in self.episodes:
            annotation = self.meta.annotations[ep_id]
            skills = sorted(
                annotation["skill_annotation"],
                key=lambda s: s["skill_idx"],
            )
            if not skills or [s["skill_idx"] for s in skills] != list(
                range(len(skills))
            ):
                raise ValueError(f"Invalid skill sequence in episode {ep_id}.")
            if any(not self.skill_labels.get(i) for i in range(len(skills))):
                raise ValueError(f"Missing configured skill label for episode {ep_id}.")
            starts = [s["frame_duration"][0] for s in skills]
            ends = [s["frame_duration"][1] for s in skills]
            valid = annotation["meta_data"]["valid_duration"]
            valid_start, valid_end = valid[0], valid[1]
            n = len(skills)
            eff_s = list(starts)
            eff_e = list(ends)
            for i in range(n):
                # Left boundary: absorb a true gap (enable_gap) or extend a
                # contiguous boundary by allow_left frames.
                if i > 0 and ends[i - 1] < starts[i]:
                    if self.enable_gap:
                        eff_s[i] = ends[i - 1]
                else:
                    eff_s[i] = starts[i] - self.allow_left
                # Right boundary: symmetric.
                if i < n - 1 and ends[i] < starts[i + 1]:
                    if self.enable_gap:
                        eff_e[i] = starts[i + 1]
                else:
                    eff_e[i] = ends[i] + self.allow_right
                eff_s[i] = max(eff_s[i], valid_start)
                eff_e[i] = min(eff_e[i], valid_end)
            self.skill_start_frames[ep_id] = eff_s
            self.skill_end_frames[ep_id] = eff_e
            if self.enable_gap:
                # Mark eligible runs before partitioning chunks, so each worker
                # only streams frames covered by at least one skill window.
                begin, end = self.episode_bounds[ep_id]
                for start, stop in sorted(zip(eff_s, eff_e)):
                    start, stop = max(begin, begin + start), min(end, begin + stop)
                    if start >= stop:
                        continue
                    if (
                        eligible
                        and eligible[-1][0] >= begin
                        and start <= eligible[-1][1]
                    ):
                        eligible[-1] = (eligible[-1][0], max(stop, eligible[-1][1]))
                    else:
                        eligible.append((start, stop))
        if self.enable_gap:
            if not eligible:
                raise ValueError(
                    "No selected frames fall inside BEHAVIOR skill windows."
                )
            self.frame_ranges = eligible

    def _is_gap_frame(self, ep_idx: int, frame_index: int) -> bool:
        """Return True if ``frame_index`` falls outside every effective window."""
        start_frames = self.skill_start_frames.get(ep_idx)
        end_frames = self.skill_end_frames.get(ep_idx)
        if start_frames is None or end_frames is None:
            return False
        for start, end in zip(start_frames, end_frames):
            if start <= frame_index < end:
                return False
        return True

    def _get_skill_label(self, item: dict) -> str:
        """Resolve a frame to a skill label, aligned with openpi-comet.

        A frame may fall inside more than one skill's effective window (overlap
        from gap absorption or boundary extension); in that case one candidate
        is chosen at random per sample. A frame inside exactly one window takes
        that skill; a frame outside all windows falls back to the nearest
        preceding skill.
        """
        ep_idx = item["episode_index"].item()
        frame_index = round(item["timestamp"].item() * self.fps)
        start_frames = self.skill_start_frames[ep_idx]
        end_frames = self.skill_end_frames[ep_idx]
        candidates = [
            i
            for i in range(len(start_frames))
            if start_frames[i] <= frame_index < end_frames[i]
        ]
        if len(candidates) == 1:
            return self.skill_labels[candidates[0]]
        if len(candidates) > 1:
            return self.skill_labels[random.choice(candidates)]
        skill_idx = bisect.bisect_right(start_frames, frame_index) - 1
        return self.skill_labels[max(0, skill_idx)]

    # -------------------------------------------------------------------------

    def get_episodes_file_paths(self) -> list[str]:
        """Return unique data, metadata and selected video paths for the chosen episodes."""
        fpaths = [str(path) for path in self._data_file_paths]
        if self.skill_labels is not None:
            fpaths.extend(
                str(self.meta.get_annotation_path(ep_idx)) for ep_idx in self.episodes
            )
        if self.fine_grained_level or self.skill_list != ["all"]:
            fpaths.extend(
                str(self.meta.get_metainfo_path(ep_idx))
                for ep_idx in self.episodes
                if self.meta.selected_episodes[ep_idx].get("metainfo_path")
            )
        for vid_key in self.meta.video_keys:
            # Several episodes share a video file in v3; only resolve it once.
            files = self._episode_rows.drop_duplicates(
                [f"videos/{vid_key}/chunk_index", f"videos/{vid_key}/file_index"]
            )
            fpaths.extend(
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for ep_idx in files.episode_index
            )
        return list(dict.fromkeys(fpaths))

    def download_episodes(self, download_videos: bool = True) -> None:
        """Download only the selected tasks' data (and videos) from the hub."""
        files = self.get_episodes_file_paths()
        if not download_videos:
            files = [path for path in files if not path.startswith("videos/")]
        self.pull_from_repo(allow_patterns=files)

    def pull_from_repo(self, allow_patterns=None, ignore_patterns=None):
        """Snapshot-download the dataset repo into ``self.root``."""
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            max_workers=max(1, os.cpu_count() - 2),
        )

    def load_hf_dataset(self):
        """Load the parquet frames for the selected episodes as a HF dataset."""
        data = Dataset.from_parquet(
            [str(self.root / path) for path in self._data_file_paths],
            features=get_hf_features_from_features(self.features),
            filters=[("episode_index", "in", self.episodes)],
        )
        data.set_transform(hf_transform_to_torch)
        return data

    def _select_streaming_chunk(self) -> None:
        """Pick this ``(rank, worker)``'s active chunk set and starting frame.

        Reads the distributed ``rank`` / ``world_size`` and the data-loader
        ``worker_id`` / ``num_workers``, folds them into the keyframe-chunk
        partition (via :func:`partition_chunk_indices`), shuffles the resulting
        chunks with a per-``(rank, worker)`` seed, and sets the streaming cursor
        to the start of the chosen chunk. Folding the rank in is what makes each
        distributed rank stream a disjoint set of chunks (a ``DistributedSampler``
        cannot, since this dataset ignores ``idx``).
        """
        # Prefer the explicit rank/world_size captured at construction (in the main
        # process); spawned DataLoader workers cannot read ``torch.distributed``, so
        # without these every rank would replicate rank 0's chunk partition.
        if self._dist_rank is not None and self._dist_world_size is not None:
            rank, world_size = self._dist_rank, self._dist_world_size
        else:
            rank = (
                dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            )
            world_size = (
                dist.get_world_size()
                if dist.is_available() and dist.is_initialized()
                else 1
            )
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers
        global_worker_id = rank * num_workers + worker_id
        if not hasattr(self, "_active_chunks") or self._active_chunks is None:
            indices = partition_chunk_indices(
                len(self.chunks),
                rank=rank,
                world_size=world_size,
                worker_id=worker_id,
                num_workers=num_workers,
            )
            worker_chunks = [self.chunks[i] for i in indices]
            rng = np.random.default_rng(self.seed + global_worker_id)
            if self.shuffle:
                rng.shuffle(worker_chunks)
            self._active_chunks = worker_chunks
            if self.skill_list != ["all"]:
                self._active_frame_count = sum(
                    end - start for start, end, _ in worker_chunks
                )
        if not self._active_chunks:
            raise ValueError(
                "Fewer BEHAVIOR chunks than data consumers. Reduce workers or ranks."
            )
        self.current_streaming_chunk_idx = 0
        if self.shuffle:
            rng = np.random.default_rng(self.seed + global_worker_id)
            self.current_streaming_chunk_idx = rng.integers(
                len(self._active_chunks)
            ).item()
        self.current_streaming_frame_idx = self._active_chunks[
            self.current_streaming_chunk_idx
        ][0]

    def __getstate__(self):
        # Each spawned worker opens its own video containers and chooses its own chunks.
        state = self.__dict__.copy()
        state.update(
            obs_loaders={}, _active_chunks=None, current_streaming_chunk_idx=None
        )
        return state

    def __getitem__(self, idx) -> dict:
        """Return the next streamed frame (the ``idx`` argument is ignored).

        In streaming mode the dataset maintains a per-consumer cursor over a
        disjoint slice of the keyframe chunks. The slice is selected so that:

        * each data loader worker on a rank streams a different set of chunks, and
        * each distributed rank streams a different set of chunks.

        Because the streaming dataset partitions itself here, a
        ``DistributedSampler`` has no effect on it (the sampler only reorders
        ``idx`` values, which are ignored). Folding the distributed rank into the
        chunk stride is therefore required to avoid every rank seeing identical
        data under ``torchrun``/DDP.
        """
        if not self._chunk_streaming_using_keyframe:
            if self.skill_labels is not None and self.enable_gap:
                if idx < 0:
                    idx += len(self)
                if not 0 <= idx < len(self):
                    raise IndexError(idx)
                run = bisect.bisect_right(self._range_ends, idx)
                idx = self.frame_ranges[run][1] - (self._range_ends[run] - idx)
            item = super().__getitem__(idx)
            self._set_prompt(item)
            return item

        if self.current_streaming_chunk_idx is None:
            self._select_streaming_chunk()

        skipped = 0
        while True:
            if (
                self.current_streaming_frame_idx
                >= self._active_chunks[self.current_streaming_chunk_idx][1]
            ):
                self.current_streaming_chunk_idx += 1
                if self.current_streaming_chunk_idx >= len(self._active_chunks):
                    self.current_streaming_chunk_idx = 0
                self.current_streaming_frame_idx = self._active_chunks[
                    self.current_streaming_chunk_idx
                ][0]
                for loader in self.obs_loaders.values():
                    loader.close()
                self.obs_loaders.clear()
            idx = self.current_streaming_frame_idx
            self.current_streaming_frame_idx += 1
            if self.skill_list == ["all"]:
                break
            frame = self.hf_dataset[idx]
            weight = skill_weight(self._get_current_task_skill(frame), self.skill_list)
            if random.choices([True, False], weights=[weight, 1 - weight])[0]:
                break
            skipped = skipped + 1 if weight == 0 else 0
            if skipped and skipped == self._active_frame_count:
                raise ValueError(
                    "No frames in this worker's chunks have positive skill sampling weight."
                )

        item = super().__getitem__(idx)
        self._set_prompt(item)
        return item

    def __len__(self):
        return self._range_ends[-1]

    def _get_current_task_skill(self, item: dict) -> str:
        """Look up the level-1 (skill) task text for the current frame."""
        ep_idx = item["episode_index"].item()
        frame_index = round(item["timestamp"].item() * self.fps)
        stages = self.meta.orchestrators[ep_idx][1]
        sub_idx = bisect.bisect_left(
            [stage["end_frame"] for stage in stages], frame_index
        )
        task_skill = stages[min(sub_idx, len(stages) - 1)]["task"]
        return task_skill

    def _get_fine_grained_task(self, item: dict) -> str:
        """Look up the fine-grained task text (prompt) for the current frame."""
        ep_idx = item["episode_index"].item()
        task_idx = item["task_index"].item()
        frame_index = round(item["timestamp"].item() * self.fps)
        name = self.meta.tasks.iloc[task_idx].name
        if ep_idx not in self.task_sizes:
            return self.task_prompts[name]
        try:
            sub_idx = min(
                bisect.bisect_left(self.task_sizes[ep_idx], frame_index),
                len(self.task_sizes[ep_idx]) - 1,
            )
            task_text = self.meta.orchestrators[ep_idx][self.fine_grained_level][
                sub_idx
            ]["task"]
        except Exception as e:
            logger.warning("%s failed to get subtask %s: %s", self.repo_id, item, e)
            task_text = self.task_prompts[name]
        return task_text

    def _set_prompt(self, item: dict) -> None:
        """Set the per-frame training-prompt fields on ``item``.

        Always sets ``item["task"]`` (the fine-grained main-task text). In skill mode
        it also resolves the per-frame skill text (``item["skill_label"]``) and, when
        ``use_skill`` is on, makes that skill text the training prompt
        (``item["prompt"]``, which the transform prefers over ``item["task"]``). With
        ``use_skill`` off, no ``prompt`` key is set, so the prompt is the task text.
        """
        item["task"] = self._get_fine_grained_task(item)
        if self.skill_labels is not None:
            item["skill_label"] = self._get_skill_label(item)
            if self.use_skill:
                item["prompt"] = item["skill_label"]

    def _get_query_indices(self, idx: int, ep_idx: int):
        """Compute action-horizon query indices and per-key padding masks."""
        ep_idx_pos = self.episode_data_index_pos[ep_idx]
        ep_start = self.episode_data_index["from"][ep_idx_pos]
        ep_end = self.episode_data_index["to"][ep_idx_pos]
        query_indices = {
            key: [
                max(ep_start.item(), min(ep_end.item() - 1, idx + delta))
                for delta in delta_idx
            ]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": th.BoolTensor(
                [
                    (idx + delta < ep_start.item()) | (idx + delta >= ep_end.item())
                    for delta in delta_idx
                ]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _query_videos(self, query_timestamps, ep_idx):
        """Decode timestamps relative to an episode's start in its shared video files."""
        if self.video_backend != "pyav":
            if "depth_linear" in self.meta.modalities:
                raise ValueError(
                    "BEHAVIOR 12-bit depth videos require the pyav backend."
                )
            return super()._query_videos(query_timestamps, ep_idx)

        item = {}
        episode = self.meta.episodes[ep_idx]
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            offset = episode[f"videos/{vid_key}/from_timestamp"]
            times = [offset + ts for ts in query_ts]
            stream = self._chunk_streaming_using_keyframe and vid_key not in (
                self.delta_indices or {}
            )
            frames = self.obs_loaders.get(vid_key) if stream else None
            if frames is None:
                frames = _stream_video_frames(video_path, min(times))
                if stream:
                    self.obs_loaders[vid_key] = frames
            try:
                decoded = {}
                for ts in sorted(set(times)):
                    for frame in frames:
                        if frame.time < ts - self.tolerance_s:
                            continue
                        if abs(frame.time - ts) <= self.tolerance_s:
                            break
                        raise ValueError(f"Missing frame at {ts:.6f}s in {video_path}.")
                    else:
                        raise ValueError(
                            f"Video ended before {ts:.6f}s in {video_path}."
                        )
                    if vid_key.startswith("observation.depth_linear."):
                        decoded[ts] = _depth_in_meters(
                            frame, self.meta.features[vid_key]["info"]
                        )
                    else:
                        rgb = frame.to_ndarray(format="rgb24")
                        decoded[ts] = (
                            th.from_numpy(rgb).permute(2, 0, 1).float().div_(255)
                        )
                item[vid_key] = th.stack([decoded[ts] for ts in times]).squeeze(0)
            finally:
                if not stream:
                    frames.close()
        return item

    def _get_keyframe_chunk_indices(self, chunk_size=250):
        """Split each episode into contiguous ``chunk_size``-frame keyframe chunks.

        Returns a list of ``(global_start, global_end, local_start)`` tuples where
        ``global_*`` index into the flat HF dataset and ``local_start`` is the
        in-episode frame offset used to seek the per-camera video loaders.
        """
        chunks = []
        ep_starts = self.episode_data_index["from"].tolist()
        for start, end in self.frame_ranges:
            offset = ep_starts[bisect.bisect_right(ep_starts, start) - 1]
            for begin in range(start, end, chunk_size):
                chunks.append((begin, min(begin + chunk_size, end), begin - offset))
        return chunks
