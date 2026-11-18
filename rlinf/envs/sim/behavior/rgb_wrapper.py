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

from __future__ import annotations

from omnigibson.envs import Environment, EnvironmentWrapper

from rlinf.envs.sim.behavior.utils import RGB_RESOLUTIONS, set_head_camera_aperture


class RGBWrapper(EnvironmentWrapper):
    """A RGB-only wrapper for OmniGibson."""

    @classmethod
    def camera_spec(cls) -> dict:
        """Declare resolutions before the SDK creates its batched cameras."""
        return {"modalities": ["rgb"], "resolution": dict(RGB_RESOLUTIONS)}

    def __init__(self, env: Environment):
        super().__init__(env=env)
        for scene in env.scenes:
            set_head_camera_aperture(scene.robots[0])
