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
import json
import random
from itertools import accumulate
from pathlib import Path

import numpy as np
import torch as th
import torch.distributed as dist
from datasets import Dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    check_delta_timestamps,
    get_delta_indices,
    get_hf_features_from_features,
    hf_transform_to_torch,
)
from torch.utils.data import get_worker_info

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
    """Read 2026 BEHAVIOR demonstrations and stream contiguous chunks per rank and worker."""

    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        episodes: list[int] | None = None,
        *,
        tasks: list[str] | None = None,
        modalities: list[str] | None = None,
        cameras: list[str] | None = None,
        skill_labels: dict[int, str] | None = None,
        use_skill: bool = False,
        enable_gap: bool = True,
        allow_left: int = 0,
        allow_right: int = 0,
        action_horizon: int | None = None,
        delta_timestamps: dict | None = None,
        tolerance_s: float = 1e-4,
        video_backend: str = "pyav",
        image_transforms=None,
        chunk_streaming_using_keyframe: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        dist_rank: int | None = None,
        dist_world_size: int | None = None,
    ):
        self.root = Path(root).expanduser()
        self.repo_id = repo_id
        with (self.root / "meta/info.json").open() as file:
            version = json.load(file)["codebase_version"]
        if version != "v3.0":
            raise ValueError(
                f"BEHAVIOR SFT requires 2026 demonstrations (v3.0), found {version!r}."
            )
        # Training uses local demonstrations. Detect incomplete metadata before LeRobot
        # attempts to fetch it from the Hub.
        for pattern in ("meta/tasks.parquet", "meta/episodes/*/*.parquet"):
            if not any(self.root.glob(pattern)):
                raise FileNotFoundError(self.root / pattern)

        self.meta = LeRobotDatasetMetadata(repo_id, self.root)
        self.revision = self.meta.revision
        self.video_backend = video_backend
        self.image_transforms = image_transforms
        self.tolerance_s = tolerance_s
        if action_horizon is not None:
            if delta_timestamps is not None:
                raise ValueError("Specify either action_horizon or delta_timestamps.")
            delta_timestamps = {"action": [i / self.fps for i in range(action_horizon)]}
        self.delta_timestamps = delta_timestamps
        self.delta_indices = None
        if delta_timestamps:
            check_delta_timestamps(delta_timestamps, self.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, self.fps)

        # Select streams before LeRobot opens their videos. The 2026 release
        # uses sensor link names for the three R1Pro cameras.
        camera_names = {
            "head": "zed_link_camera_0",
            "left_wrist": "left_realsense_link_camera_0",
            "right_wrist": "right_realsense_link_camera_0",
        }
        selected_cameras = [
            camera_names.get(cam, cam) for cam in cameras or camera_names
        ]
        if set(modalities or ["rgb"]) != {"rgb"}:
            raise ValueError(
                "The BEHAVIOR SFT LeRobot reader supports RGB videos only. "
                "2026 depth videos require a 12-bit depth decoder."
            )
        selected_features = {f"observation.rgb.{cam}" for cam in selected_cameras}
        video_keys = set(self.meta.video_keys)
        if video_keys or cameras or modalities:
            missing = selected_features - video_keys
            if missing:
                raise ValueError(f"Missing BEHAVIOR camera features: {sorted(missing)}")
        for key in video_keys - selected_features:
            del self.meta.features[key]
        with (self.root / "meta/tasks.jsonl").open() as file:
            self.task_prompts = {
                row["task_name"]: row["task"] for row in map(json.loads, file)
            }

        self.task_names = set(tasks or ())
        self._episode_positions = episodes
        self.skill_labels = skill_labels
        self.use_skill = use_skill
        self.enable_gap = enable_gap
        self.allow_left = allow_left
        self.allow_right = allow_right
        if use_skill and not skill_labels:
            raise ValueError("BEHAVIOR use_skill=True requires explicit skill_labels.")
        self.hf_dataset = self.load_hf_dataset()
        self.frame_ranges = list(self.episode_bounds.values())
        if skill_labels is not None:
            self._build_skill_boundaries()
        self._range_ends = list(
            accumulate(end - start for start, end in self.frame_ranges)
        )
        self._chunk_streaming_using_keyframe = chunk_streaming_using_keyframe
        self.shuffle = shuffle
        self.seed = seed
        self._dist_rank = dist_rank
        self._dist_world_size = dist_world_size
        if chunk_streaming_using_keyframe:
            self.chunks = self._get_keyframe_chunk_indices()
            self.current_streaming_chunk_idx = None

    def load_hf_dataset(self):
        """Read the Parquet shards for the selected episodes in published frame order."""
        columns = [
            "episode_index",
            "tasks",
            "length",
            "data/chunk_index",
            "data/file_index",
        ]
        if self.skill_labels is not None:
            columns.append("annotation_path")
        rows = self.meta.episodes.select_columns(columns).to_pandas()
        selected = rows
        if self.task_names or self._episode_positions is not None:
            by_task = rows[["episode_index", "tasks"]].explode("tasks")
            if self.task_names:
                by_task = by_task[by_task.tasks.isin(self.task_names)]
            if self._episode_positions is not None:
                # Episode positions are relative to each selected task.
                by_task = by_task.groupby("tasks", sort=False).nth(
                    self._episode_positions
                )
            selected = rows[rows.episode_index.isin(by_task.episode_index)]
        if selected.empty:
            raise ValueError(
                f"No BEHAVIOR 2026 episodes match {sorted(self.task_names)}."
            )
        self.episodes = selected.episode_index.tolist()
        if self.skill_labels is not None:
            self.annotation_paths = dict(zip(self.episodes, selected.annotation_path))
        shards = selected.drop_duplicates(["data/chunk_index", "data/file_index"])
        paths = [
            self.root / self.meta.get_data_file_path(i) for i in shards.episode_index
        ]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        data = Dataset.from_parquet(
            [str(path) for path in paths],
            features=get_hf_features_from_features(self.features),
            filters=[("episode_index", "in", self.episodes)],
        )

        # LeRobot v3 stores global offsets. A local selection starts at zero,
        # so keep the action window within the selected episode's local bounds.
        ends = selected.length.cumsum()
        starts = ends - selected.length
        if len(data) != ends.iloc[-1] or (
            data.data.column("episode_index").take(starts.tolist()).to_pylist()
            != self.episodes
            or data.data.column("episode_index").take((ends - 1).tolist()).to_pylist()
            != self.episodes
        ):
            raise ValueError(
                "Local BEHAVIOR frames do not match the selected episode lengths."
            )
        self.episode_bounds = dict(zip(self.episodes, zip(starts, ends)))
        data.set_transform(hf_transform_to_torch)
        return data

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
        eligible = []
        for ep_id in self.episodes:
            with (self.root / self.annotation_paths[ep_id]).open() as file:
                annotation = json.load(file)
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

    def _get_skill_label(self, item: dict) -> str:
        """Resolve a frame to a skill label, aligned with openpi-comet.

        A frame may fall inside more than one skill's effective window (overlap
        from gap absorption or boundary extension); in that case one candidate
        is chosen at random per sample. A frame inside exactly one window takes
        that skill; a frame outside all windows falls back to the nearest
        preceding skill.
        """
        ep_idx = item["episode_index"].item()
        frame_index = item["frame_index"].item()
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
        indices = partition_chunk_indices(
            len(self.chunks),
            rank=rank,
            world_size=world_size,
            worker_id=worker_id,
            num_workers=num_workers,
        )
        self._active_chunks = [self.chunks[i] for i in indices]
        if not self._active_chunks:
            raise ValueError(
                "Fewer BEHAVIOR chunks than data consumers. Reduce data.num_workers or the number of ranks."
            )
        self.current_streaming_chunk_idx = 0
        if self.shuffle:
            rng = np.random.default_rng(self.seed + global_worker_id)
            rng.shuffle(self._active_chunks)
            self.current_streaming_chunk_idx = rng.integers(
                len(self._active_chunks)
            ).item()
        self.current_streaming_frame_idx = self._active_chunks[
            self.current_streaming_chunk_idx
        ][0]

    def __getitem__(self, idx) -> dict:
        if self._chunk_streaming_using_keyframe:
            if self.current_streaming_chunk_idx is None:
                self._select_streaming_chunk()
            if (
                self.current_streaming_frame_idx
                >= self._active_chunks[self.current_streaming_chunk_idx][1]
            ):
                self.current_streaming_chunk_idx = (
                    self.current_streaming_chunk_idx + 1
                ) % len(self._active_chunks)
                self.current_streaming_frame_idx = self._active_chunks[
                    self.current_streaming_chunk_idx
                ][0]
            idx = self.current_streaming_frame_idx
            self.current_streaming_frame_idx += 1

        elif self.skill_labels is not None and self.enable_gap:
            if idx < 0:
                idx += len(self)
            if not 0 <= idx < len(self):
                raise IndexError(idx)
            run = bisect.bisect_right(self._range_ends, idx)
            idx = self.frame_ranges[run][1] - (self._range_ends[run] - idx)

        item = super().__getitem__(idx)
        self._set_prompt(item)
        return item

    def __len__(self):
        return self._range_ends[-1]

    def _set_prompt(self, item: dict) -> None:
        """Set the per-frame training-prompt fields on ``item``.

        Always sets ``item["task"]`` (the fine-grained main-task text). In skill mode
        it also resolves the per-frame skill text (``item["skill_label"]``) and, when
        ``use_skill`` is on, makes that skill text the training prompt
        (``item["prompt"]``, which the transform prefers over ``item["task"]``). With
        ``use_skill`` off, no ``prompt`` key is set, so the prompt is the task text.
        """
        item["task"] = self.task_prompts[item["task"]]
        if self.skill_labels is not None:
            item["skill_label"] = self._get_skill_label(item)
            if self.use_skill:
                item["prompt"] = item["skill_label"]

    def _get_query_indices(self, idx: int, ep_idx: int):
        """Compute action-horizon query indices and per-key padding masks."""
        ep_start, ep_end = self.episode_bounds[ep_idx]
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": th.BoolTensor(
                [
                    (idx + delta < ep_start) | (idx + delta >= ep_end)
                    for delta in delta_idx
                ]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_keyframe_chunk_indices(self, chunk_size=250):
        """Split each selected episode into contiguous chunks without crossing its boundary."""
        return [
            (start, min(start + chunk_size, end))
            for begin, end in self.frame_ranges
            for start in range(begin, end, chunk_size)
        ]
