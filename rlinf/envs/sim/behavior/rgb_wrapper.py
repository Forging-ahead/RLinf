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

from rlinf.envs.sim.behavior.utils import CAMERA_LINKS


class RGBWrapper(EnvironmentWrapper):
    """A RGB-only wrapper for OmniGibson."""

    def __init__(self, env: Environment):
        super().__init__(env=env)
        for scene in env.scenes:
            robot = scene.robots[0]
            sensor_name = f"{robot.name}:{CAMERA_LINKS['head']}"
            robot.sensors[sensor_name].horizontal_aperture = 40.0
