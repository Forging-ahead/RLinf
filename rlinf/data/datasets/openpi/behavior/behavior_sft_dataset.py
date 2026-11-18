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

import json
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

        # The 2026 metadata describes RGB and depth. OpenPI consumes only RGB.
        for key in self.meta.video_keys:
            if key.startswith("observation.depth_linear."):
                del self.meta.features[key]
        with (self.root / "meta/tasks.jsonl").open() as file:
            self.task_prompts = {
                row["task_name"]: row["task"] for row in map(json.loads, file)
            }

        self.task_names = set(tasks or ())
        self._episode_positions = episodes
        self.hf_dataset = self.load_hf_dataset()
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

        item = super().__getitem__(idx)
        item["task"] = self.task_prompts[item["task"]]
        return item

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
            for begin, end in self.episode_bounds.values()
            for start in range(begin, end, chunk_size)
        ]
