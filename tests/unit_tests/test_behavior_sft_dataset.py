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

import numpy as np
import pytest

pytest.importorskip("openpi")

from rlinf.data.datasets.openpi.behavior.behavior_sft_data_loader import _Repack
from rlinf.models.embodiment.openpi.policies.behavior_policy import (
    extract_state_from_proprio,
)


@pytest.mark.parametrize("year", [2025, 2026])
def test_behavior_sft_repack_preserves_camera_order_state_and_task(year):
    prompt = "Turn on the radio receiver that's on the table in the living room."
    if year == 2026:
        keys = (
            "observation.rgb.zed_link_camera_0",
            "observation.rgb.left_realsense_link_camera_0",
            "observation.rgb.right_realsense_link_camera_0",
        )
        state = np.arange(61, dtype=np.float32)
        frame = {"task": "turning_on_radio"}
        repack = _Repack(keys, {"turning_on_radio": prompt})
    else:
        keys = (
            "observation.images.rgb.head",
            "observation.images.rgb.left_wrist",
            "observation.images.rgb.right_wrist",
        )
        state = np.arange(256, dtype=np.float32)
        frame = {"prompt": prompt}
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
