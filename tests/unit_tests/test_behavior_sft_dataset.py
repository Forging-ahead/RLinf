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

"""BEHAVIOR camera/state/prompt inputs at the shared OpenPI SFT boundary."""

import json

import numpy as np
import pytest

pytest.importorskip("openpi")

from rlinf.data.datasets.openpi.behavior import BehaviorSftDataset
from rlinf.data.datasets.openpi.behavior.behavior_sft_data_loader import _Repack
from rlinf.models.embodiment.openpi.policies.behavior_policy import (
    extract_state_from_proprio,
)


def test_behavior_sft_repack_preserves_camera_order_state_and_task():
    prompt = "Turn on the radio receiver that's on the table in the living room."
    keys = (
        "observation.rgb.zed_link_camera_0",
        "observation.rgb.left_realsense_link_camera_0",
        "observation.rgb.right_realsense_link_camera_0",
    )
    state = np.arange(61, dtype=np.float32)
    frame = {"task": prompt}
    repack = _Repack()
    for index, key in enumerate(keys):
        frame[key] = np.full((3, 4, 4), index, dtype=np.uint8)
    frame["observation.state"] = state
    frame["action"] = np.arange(32 * 23, dtype=np.float32).reshape(32, 23)

    item = repack(frame)
    assert item["prompt"] == prompt
    np.testing.assert_array_equal(item["observation/state"], state)
    np.testing.assert_array_equal(item["actions"], frame["action"])
    np.testing.assert_array_equal(item["observation/image"], frame[keys[0]])
    np.testing.assert_array_equal(item["observation/wrist_image"][0], frame[keys[1]])
    np.testing.assert_array_equal(item["observation/wrist_image"][1], frame[keys[2]])


@pytest.mark.parametrize(
    ("dimension", "indices"),
    [
        (
            61,
            [
                *range(3),
                *range(53, 57),
                *range(3, 10),
                *range(28, 35),
                24 + 25,
                49 + 50,
            ],
        ),
        (
            256,
            [
                *range(253, 256),
                *range(236, 240),
                *range(158, 165),
                *range(197, 204),
                193 + 194,
                232 + 233,
            ],
        ),
    ],
)
def test_behavior_policy_maps_both_r1pro_layouts(dimension, indices):
    proprio = np.arange(dimension, dtype=np.float32)
    np.testing.assert_array_equal(extract_state_from_proprio(proprio), indices)


@pytest.fixture
def shared_shard(tmp_path):
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

    if CODEBASE_VERSION != "v3.0":
        pytest.skip("Requires the 2026 BEHAVIOR SFT LeRobot v3 reader")

    root = tmp_path / "demonstrations"
    feature = {"dtype": "float32", "shape": (2,), "names": None}
    writer = LeRobotDataset.create(
        "local/test",
        fps=30,
        root=root,
        features={"action": feature, "observation.state": feature},
        use_videos=False,
    )
    for task, start in (("other_task", 0), ("turning_on_radio", 10)):
        for value in (start, start + 1):
            action = np.array([value, -value], dtype=np.float32)
            writer.add_frame(
                {"task": task, "action": action, "observation.state": action}
            )
        writer.save_episode()
    assert len(list((root / "data").rglob("*.parquet"))) == 1
    (root / "meta/tasks.jsonl").write_text(
        '{"task_name":"turning_on_radio","task":"Turn on the radio."}\n'
    )
    return root


def test_behavior_sft_selects_task_from_shared_shard(shared_shard):
    dataset = BehaviorSftDataset(
        "local/test",
        shared_shard,
        tasks=["turning_on_radio"],
        action_horizon=3,
        shuffle=False,
    )
    assert len(dataset) == 2
    first, last = dataset[0], dataset[1]
    assert int(first["episode_index"]) == 1
    assert first["task"] == "Turn on the radio."
    np.testing.assert_array_equal(first["action"], [[10, -10], [11, -11], [11, -11]])
    np.testing.assert_array_equal(last["action"], [[11, -11]] * 3)
    np.testing.assert_array_equal(last["action_is_pad"], [False, True, True])


@pytest.mark.parametrize("streaming", [False, True])
def test_behavior_skill_uses_episode_annotation_and_preserves_actions(
    shared_shard, streaming
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    (metadata_file,) = (shared_shard / "meta/episodes").rglob("*.parquet")
    metadata = pq.read_table(metadata_file)
    annotation_path = "annotations/task-0000/episode_00000020.json"
    pq.write_table(
        metadata.append_column(
            "annotation_path", pa.array(["unused.json", annotation_path])
        ),
        metadata_file,
    )
    kwargs = {
        "tasks": ["turning_on_radio"],
        "action_horizon": 3,
        "chunk_streaming_using_keyframe": streaming,
        "shuffle": False,
        "use_skill": True,
        "skill_labels": {0: "move to radio", 1: "press radio"},
    }
    with pytest.raises(FileNotFoundError, match="episode_00000020.json"):
        BehaviorSftDataset("local/test", shared_shard, **kwargs)
    path = shared_shard / annotation_path
    path.parent.mkdir(parents=True)
    annotation = {
        "meta_data": {"valid_duration": [0, 2]},
        "skill_annotation": [
            {"skill_idx": 0, "frame_duration": [0, 1]},
            {"skill_idx": 1, "frame_duration": [1, 2]},
        ],
    }
    path.write_text(json.dumps(annotation))
    dataset = BehaviorSftDataset("local/test", shared_shard, **kwargs)
    assert len(dataset) == 2
    first, last = dataset[0], dataset[1]
    assert first["task"] == last["task"] == "Turn on the radio."
    assert first["prompt"] == "move to radio"
    assert last["prompt"] == "press radio"
    np.testing.assert_array_equal(first["action"], [[10, -10], [11, -11], [11, -11]])
    np.testing.assert_array_equal(last["action_is_pad"], [False, True, True])

    annotation["meta_data"]["valid_duration"] = [1, 2]
    path.write_text(json.dumps(annotation))
    dataset = BehaviorSftDataset("local/test", shared_shard, **kwargs)
    assert len(dataset) == 1
    assert int(dataset[0]["frame_index"]) == 1
    assert dataset[0]["prompt"] == "press radio"


def test_behavior_sft_rejects_v21(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text('{"codebase_version":"v2.1"}')
    with pytest.raises(ValueError, match=r"2026 demonstrations .*v2\.1"):
        BehaviorSftDataset("local/test", tmp_path)
