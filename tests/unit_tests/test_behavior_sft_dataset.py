# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Offline BEHAVIOR demonstration decoding in the OpenPI model environment."""

import numpy as np
import pytest
import torch

av = pytest.importorskip("av")
pytest.importorskip("openpi")
pytest.importorskip("lerobot")


def test_preserves_recorded_timestamps_and_quantiles():
    from rlinf.data.datasets.openpi.behavior.behavior_sft_dataset import (
        _episode_stats,
        _frame_tensors,
    )

    timestamp = 1955 / 30
    item = _frame_tensors({"timestamp": [timestamp], "action": [[1.0, 2.0]]})
    assert item["timestamp"][0].dtype == torch.float64
    assert item["timestamp"][0].item() == timestamp
    torch.testing.assert_close(item["action"][0], torch.tensor([1.0, 2.0]))

    def episode(value):
        return {
            "action": {
                key: np.array([v])
                for key, v in {
                    "min": value,
                    "max": value,
                    "mean": value,
                    "std": 0.0,
                    "count": 1,
                    "q01": value,
                    "q99": value,
                }.items()
            }
        }

    stats = _episode_stats([episode(0.0), episode(10.0)])["action"]
    np.testing.assert_allclose(stats["mean"], [5.0])
    np.testing.assert_allclose(stats["q01"], [0.1])
    np.testing.assert_allclose(stats["q99"], [9.9])


@pytest.mark.parametrize("modality", ["rgb", "depth", "seg_instance_id"])
def test_stream_seeks_to_requested_frame_and_decodes_2025_modalities(
    tmp_path, modality
):
    from rlinf.data.datasets.openpi.behavior.behavior_sft_dataset import _video_frames

    path = tmp_path / "demo.mkv"
    is_depth = modality == "depth"
    fmt = "gray16le" if is_depth else "rgb24"
    with av.open(str(path), "w") as writer:
        stream = writer.add_stream("ffv1", rate=25)
        stream.width = stream.height = 8
        stream.pix_fmt = "gray16le" if is_depth else "bgr0"
        for i in range(2):
            if is_depth:
                pixels = np.full((8, 8), (2**14 - 1) * i, dtype=np.uint16)
            else:
                pixels = np.full(
                    (8, 8, 3), [16, 16, 16 if i == 0 else 240], dtype=np.uint8
                )
            writer.mux(stream.encode(av.VideoFrame.from_ndarray(pixels, format=fmt)))
        writer.mux(stream.encode())

    frames = list(_video_frames(path, modality, start=1, fps=25, instance_ids=[3, 55]))
    assert len(frames) == 1
    if is_depth:
        torch.testing.assert_close(frames[0], torch.full((8, 8), 10.0))
    elif modality == "seg_instance_id":
        torch.testing.assert_close(frames[0], torch.full((8, 8), 55))
    else:
        assert frames[0].shape == (3, 8, 8)
        torch.testing.assert_close(
            frames[0][:, 0, 0], torch.tensor([16, 16, 240], dtype=torch.uint8)
        )
