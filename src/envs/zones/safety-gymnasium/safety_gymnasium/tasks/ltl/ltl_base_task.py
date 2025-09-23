# Copyright 2024 anonymous-elephant. All Rights Reserved.
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
# ==============================================================================
"""LTL level 0."""

from safety_gymnasium.assets.geoms import LtlWalls
from safety_gymnasium.bases.base_task import BaseTask


class LtlBaseTask(BaseTask):
    """Base task for LTL tasks."""

    def __init__(self, config, zone_size: float, walls=True) -> None:
        super().__init__(config=config)
        self.zone_size = zone_size
        self.placements_conf.extents = [-2.5, -2.5, 2.5, 2.5]
        self.lidar_conf.num_bins = 16
        self.lidar_conf.max_dist = None
        self.lidar_conf.exp_gain = 0.5
        self.lidar_conf.alias = True
        self.cost_conf.constrain_indicator = False
        self.observation_flatten = False
        if walls:
            self._add_geoms(LtlWalls())

    def calculate_reward(self):
        return 0

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def update_world(self):
        pass

    @property
    def goal_achieved(self):
        return False
