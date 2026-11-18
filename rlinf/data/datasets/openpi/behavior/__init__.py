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

from rlinf.data.datasets.openpi.behavior.behavior_sft_data_loader import (
    BehaviorSftDataConfig,
    BehaviorSftDataLoader,
    build_behavior_sft_dataloader,
    create_behavior_sft_data_loader,
)

__all__ = [
    "BehaviorSftDataConfig",
    "BehaviorSftDataLoader",
    "BehaviorSftDataset",
    "build_behavior_sft_dataloader",
    "create_behavior_sft_data_loader",
]


def __getattr__(name):
    if name == "BehaviorSftDataset":
        # The 2025 streaming dataset imports the LeRobot v2.1 API.
        from rlinf.data.datasets.openpi.behavior.behavior_sft_dataset import (
            BehaviorSftDataset,
        )

        return BehaviorSftDataset
    raise AttributeError(name)
