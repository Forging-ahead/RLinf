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

import dataclasses
import json
import logging
import multiprocessing
import typing
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from rlinf.data.datasets.openpi.transform_fn import DataTransformFn, compose
from rlinf.data.storage.lerobot import (
    resolve_lerobot_repo_id,
)
from rlinf.models.embodiment.openpi.modules.model import Observation
from rlinf.models.embodiment.openpi.transforms.pipeline import (
    build_openpi_transforms,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BehaviorSftDataConfig",
    "BehaviorSftDataLoader",
    "build_behavior_sft_dataloader",
    "create_behavior_sft_data_loader",
]

# Camera views resolved by the BEHAVIOR pi05 transform.
_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


@dataclasses.dataclass(frozen=True)
class _Repack(DataTransformFn):
    """Map 2026 LeRobot frame keys to the ``observation/*`` names openpi's
    ``BehaviorInputs`` expects (a ``DataTransformFn``, so it composes
    directly in front of the shared pipeline).

    The two wrist views are stacked into a single ``observation/wrist_image``
    ``[2, C, H, W]`` array in ``(left, right)`` order — the layout openpi's
    ``BehaviorInputs`` consumes (it splits index ``0``/``1`` into the left/right
    wrist). Images stay ``(C, H, W)`` float; ``BehaviorInputs._parse_image`` does
    the channel reorder and the float→uint8 conversion.
    """

    task_prompts: dict[str, str]

    def __call__(self, frame: dict) -> dict:
        return {
            "observation/image": np.asarray(frame["observation.rgb.zed_link_camera_0"]),
            "observation/wrist_image": np.stack(
                [
                    frame["observation.rgb.left_realsense_link_camera_0"],
                    frame["observation.rgb.right_realsense_link_camera_0"],
                ],
                axis=0,
            ),
            "observation/state": np.asarray(frame["observation.state"]),
            "actions": np.asarray(frame["action"]),
            "prompt": self.task_prompts[frame["task"]],
        }


class _LocalBehaviorLeRobotDataset(LeRobotDataset):
    def _check_cached_episodes_sufficient(self) -> bool:
        # Read local episode IDs in Arrow without converting every frame ID to a tensor.
        self.local_episodes = self.hf_dataset.unique("episode_index")
        if not self.local_episodes:
            raise ValueError(f"No local BEHAVIOR frames under {self.root}.")
        return True


class _TransformedStreamingDataset(torch.utils.data.Dataset):
    """Apply the OpenPI input transform to each selected LeRobot frame."""

    def __init__(self, dataset, transform):
        self._dataset = dataset
        self._transform = transform

    def __getitem__(self, idx):
        return self._transform(self._dataset[idx])

    def __len__(self) -> int:
        return len(self._dataset)


def _sft_collate(items) -> tuple[Observation, torch.Tensor]:
    """Collate per-sample transform dicts into ``(Observation, actions)``.

    Plain numpy/torch stacking (no JAX): each per-sample dict is the openpi
    model-input format (nested ``image`` / ``image_mask`` dicts, plus ``state`` /
    ``actions`` / ``tokenized_prompt`` / ``tokenized_prompt_mask``). State and
    actions are cast to float32 (the verified SFT boundary dtype);
    ``Observation.from_dict`` converts the ``uint8`` images to ``float32`` in
    ``[-1, 1]``. Returned actions have shape ``[batch, action_horizon, action_dim]``.
    """
    if not items:
        raise ValueError("Cannot collate an empty BEHAVIOR SFT batch.")

    def _stack(key, dtype=None):
        return torch.from_numpy(
            np.stack([np.asarray(item[key], dtype=dtype) for item in items])
        )

    images = {
        key: torch.from_numpy(
            np.stack([np.asarray(item["image"][key]) for item in items])
        )
        for key in _IMAGE_KEYS
    }
    image_masks = {
        key: torch.from_numpy(
            np.stack(
                [np.asarray(item["image_mask"][key], dtype=np.bool_) for item in items]
            )
        )
        for key in _IMAGE_KEYS
    }
    observation = Observation.from_dict(
        {
            "image": images,
            "image_mask": image_masks,
            "state": _stack("state", np.float32),
            "tokenized_prompt": _stack("tokenized_prompt", np.int64).long(),
            "tokenized_prompt_mask": _stack("tokenized_prompt_mask", np.bool_),
        }
    )
    actions = _stack("actions", np.float32)
    return observation, actions


@dataclasses.dataclass(frozen=True)
class BehaviorSftDataConfig:
    """Resolved BEHAVIOR SFT data-pipeline metadata, exposed via
    :meth:`BehaviorSftDataLoader.data_config`."""

    repo_id: str
    action_dim: int
    action_horizon: int
    max_token_len: int


def _worker_init_fn(worker_id: int) -> None:
    """Per-worker init hook (placeholder for worker-local environment setup)."""
    del worker_id


def create_behavior_sft_data_loader(
    *,
    behavior_dataset_root: str,
    model_path: str,
    config_name: str,
    repo_id: str,
    tasks: list[str],
    action_dim: int,
    action_horizon: int,
    max_token_len: int,
    batch_size: int,
    num_workers: int,
    tolerance_s: float,
    shuffle: bool,
    seed: int,
    dist_rank: int,
    dist_world_size: int,
    data_kwargs: dict | None = None,
) -> "BehaviorSftDataLoader":
    """Build the BEHAVIOR-1K SFT data loader yielding ``(Observation, actions)``.

    Args:
        behavior_dataset_root: Local root of the LeRobot BEHAVIOR dataset.
        model_path: New-format checkpoint dir passed to ``get_openpi_config`` to
            select the shared transform pipeline.
        config_name: openpi TrainConfig key (e.g. ``pi05_behavior``).
        repo_id: LeRobot dataset repo id (used for metadata bookkeeping).
        tasks: BEHAVIOR task names to include.
        action_dim: Model action dimension (metadata).
        action_horizon: Number of future action steps per sample.
        max_token_len: Maximum tokenized-prompt length (metadata).
        batch_size: Per-rank batch size.
        num_workers: Number of ``DataLoader`` workers (``> 0`` uses ``spawn``).
        tolerance_s: Frame-timestamp sync tolerance in seconds.
        shuffle: Whether to shuffle the selected training frames.
        seed: Seed for shuffling and worker initialization.
        dist_rank: Rank of this training process.
        dist_world_size: Number of training ranks.
        data_kwargs: Optional ``openpi_data`` overrides forwarded to the
            pipeline (``norm_stats_path`` pins ``norm_stats.json``).

    Returns:
        A loader whose iteration yields ``(Observation, actions)`` 2-tuples.
    """
    root = Path(behavior_dataset_root).expanduser()
    with (root / "meta/info.json").open() as file:
        info = json.load(file)
    if info["codebase_version"] != "v3.0":
        raise ValueError(
            f"BEHAVIOR SFT requires 2026 demonstrations (v3.0), found {info['codebase_version']!r}."
        )
    # LeRobot tries to download the whole dataset if the local Parquet files are missing.
    for pattern in (
        "meta/tasks.parquet",
        "meta/episodes/*/*.parquet",
        "data/*/*.parquet",
    ):
        if not any(root.glob(pattern)):
            raise FileNotFoundError(root / pattern)
    action_times = {"action": [i / info["fps"] for i in range(action_horizon)]}
    native = _LocalBehaviorLeRobotDataset(
        repo_id,
        root=root,
        delta_timestamps=action_times,
        tolerance_s=tolerance_s,
        video_backend="pyav",
    )
    # OpenPI consumes only RGB. Select its streams before LeRobot decodes frames.
    for key in native.meta.video_keys:
        if key.startswith("observation.depth_linear."):
            del native.meta.features[key]

    # The initial official v3 reader addresses frames by their row number in
    # the local parquet files. Rebase offsets for a partial download so each
    # action window stays in its own episode, including its last padded frame.
    bounds = ["dataset_from_index", "dataset_to_index"]
    rows = native.meta.episodes.select_columns(
        ["episode_index", "tasks", *bounds]
    ).to_pandas()
    present = rows.episode_index.isin(native.local_episodes)
    task_set = set(tasks)
    selected = rows.tasks.map(
        lambda names: not task_set or not task_set.isdisjoint(names)
    )
    if not selected.any():
        raise ValueError(f"No BEHAVIOR 2026 episodes match {tasks}.")
    if (selected & ~present).any():
        missing = rows.loc[selected & ~present, "episode_index"].iloc[0]
        raise FileNotFoundError(f"Missing local BEHAVIOR data for episode {missing}.")
    lengths = (rows.dataset_to_index - rows.dataset_from_index).loc[present]
    ends = lengths.cumsum()
    if ends.iloc[-1] != len(native):
        raise ValueError(
            "Local parquet rows do not match the published BEHAVIOR episode lengths."
        )
    if not ends.equals(rows.dataset_to_index.loc[present]):
        rows.loc[present, "dataset_from_index"] = ends - lengths
        rows.loc[present, "dataset_to_index"] = ends
        updated = native.meta.episodes.remove_columns(bounds)
        for key in bounds:
            updated = updated.add_column(key, rows[key].tolist())
        native.meta.episodes = updated
    dataset = torch.utils.data.ConcatDataset(
        [
            torch.utils.data.Subset(native, range(start, end))
            for start, end in rows.loc[selected, bounds].itertuples(
                index=False, name=None
            )
        ]
    )
    with (root / "meta/tasks.jsonl").open() as file:
        prompts = {row["task_name"]: row["task"] for row in map(json.loads, file)}
    repack = _Repack(prompts)

    # The shared openpi input pipeline (BehaviorInputs -> Normalize -> ModelTransform),
    # keyed off the checkpoint / config_name. Norm stats come from
    # ``openpi_data.norm_stats_path`` or the OpenPI TrainConfig default.
    # Only the transform content comes from openpi; the wrapper + collate are plain
    # numpy/torch. The composed transform is built in the main process and is
    # picklable, so spawn workers receive it directly (no lazy per-worker build).
    input_transforms, _ = build_openpi_transforms(
        model_path,
        config_name,
        data_kwargs=data_kwargs,
    )
    source = _TransformedStreamingDataset(dataset, compose([repack, *input_transforms]))

    sampler = None
    if dist_world_size > 1:
        sampler = torch.utils.data.DistributedSampler(
            source,
            num_replicas=dist_world_size,
            rank=dist_rank,
            shuffle=shuffle,
            seed=seed,
        )
    mp_context = multiprocessing.get_context("spawn") if num_workers > 0 else None

    generator = torch.Generator()
    generator.manual_seed(seed)

    logger.info(
        "BEHAVIOR SFT data loader: batch_size=%d, num_workers=%d, action_horizon=%d",
        batch_size,
        num_workers,
        action_horizon,
    )

    torch_loader = torch.utils.data.DataLoader(
        typing.cast(torch.utils.data.Dataset, source),
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,
        collate_fn=_sft_collate,
        worker_init_fn=_worker_init_fn,
        drop_last=True,
        generator=generator,
    )

    data_config = BehaviorSftDataConfig(
        repo_id=repo_id,
        action_dim=action_dim,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
    )
    return BehaviorSftDataLoader(torch_loader, data_config)


class BehaviorSftDataLoader:
    """Infinite ``(Observation, actions)`` loop over the BEHAVIOR SFT dataset.

    Re-iterates the underlying ``torch`` ``DataLoader`` forever. Each batch is
    already collated into an :class:`Observation` plus an actions tensor of shape
    ``[batch, action_horizon, action_dim]`` by :func:`_sft_collate`.
    """

    def __init__(
        self,
        torch_loader: torch.utils.data.DataLoader,
        data_config: BehaviorSftDataConfig,
    ):
        self._torch_loader = torch_loader
        self._data_config = data_config
        self._epoch = 0

    def data_config(self) -> BehaviorSftDataConfig:
        """Return the resolved data-pipeline metadata."""
        return self._data_config

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        """Expose the underlying ``torch`` ``DataLoader``."""
        return self._torch_loader

    def __iter__(self):
        while True:
            if isinstance(
                self._torch_loader.sampler, torch.utils.data.DistributedSampler
            ):
                self._torch_loader.sampler.set_epoch(self._epoch)
                self._epoch += 1
            yield from self._torch_loader

    def __len__(self) -> int:
        return len(self._torch_loader)


def build_behavior_sft_dataloader(
    cfg, world_size, rank, data_paths, eval_dataset=False
):
    """Build the BEHAVIOR SFT data loader for the SFT worker.

    The sampler partitions 2026 demonstration frames across training ranks.
    Returns ``(loader, loader.data_config())``.
    """
    from omegaconf import OmegaConf

    data_path = resolve_lerobot_repo_id(data_paths)
    if data_path is None:
        raise ValueError("openpi BEHAVIOR SFT requires data.train_data_paths.")

    model_cfg = cfg.actor.model
    data_cfg = cfg.data

    config_name = str(model_cfg.openpi.config_name)
    data_kwargs = OmegaConf.select(model_cfg, "openpi_data", default=None)
    if data_kwargs is not None:
        data_kwargs = OmegaConf.to_container(data_kwargs, resolve=True)

    loader = create_behavior_sft_data_loader(
        behavior_dataset_root=str(data_cfg.behavior_dataset_root),
        model_path=str(model_cfg.model_path),
        config_name=config_name,
        repo_id=str(data_cfg.repo_id),
        tasks=list(data_cfg.tasks),
        action_dim=int(model_cfg.openpi.model_action_dim),
        action_horizon=int(model_cfg.num_action_chunks),
        max_token_len=int(model_cfg.openpi.max_token_len),
        batch_size=int(cfg.actor.eval_batch_size)
        if eval_dataset
        else int(cfg.actor.micro_batch_size),
        num_workers=int(data_cfg.num_workers),
        tolerance_s=float(data_cfg.tolerance_s),
        shuffle=not eval_dataset,
        seed=int(cfg.actor.seed),
        dist_rank=rank,
        dist_world_size=world_size,
        data_kwargs=data_kwargs,
    )
    return loader, loader.data_config()
