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

from dataclasses import dataclass

import numpy as np
from safety_gymnasium.assets.color import COLOR
from safety_gymnasium.assets.group import GROUP
from safety_gymnasium.bases.base_object import Geom


@dataclass
class LtlWalls(Geom):  # pylint: disable=too-many-instance-attributes
    """
    The walls at the boundary of any LTL tasks.
    """

    name: str = 'ltl_walls'
    num: int = 4
    locate_factor: float = 3.5
    collision_threshold: float = 3.3
    size: float = 3.5
    placements: list = None
    keepout: float = 0.0

    color: np.array = COLOR['wall']
    alpha: float = 0.9
    group: np.array = GROUP['wall']
    is_lidar_observed: bool = False

    def __post_init__(self) -> None:
        assert self.num in (2, 4)
        assert (
            self.locate_factor >= 0
        ), 'For cost calculation, the locate_factor must be greater than or equal to zero.'
        self.locations: list = [
            (self.locate_factor, 0),
            (-self.locate_factor, 0),
            (0, self.locate_factor),
            (0, -self.locate_factor),
        ]

        self.index: int = 0

    def index_tick(self):
        """Count index."""
        self.index += 1
        self.index %= self.num

    def get_config(self, xy_pos, rot):  # pylint: disable=unused-argument
        """To facilitate get specific config for this object."""
        body = {
            'name': self.name,
            'pos': np.r_[xy_pos, 0.25],
            'rot': 0,
            'geoms': [
                {
                    'name': self.name,
                    'size': np.array([0.05, self.size, 0.3]),
                    'type': 'box',
                    'contype': 0,
                    'conaffinity': 0,
                    'group': self.group,
                    'rgba': self.color * np.array([1, 1, 1, self.alpha]),
                },
            ],
        }
        if self.index >= 2:
            body.update({'rot': np.pi / 2})
        self.index_tick()
        return body

    def cal_cost(self):
        x, y, _ = list(self.agent.pos)
        cost = {
            'wall_sensor': self.wall_sensor(x, y),
            'cost_ltl_walls': 0
        }
        if x >= self.collision_threshold or x <= -self.collision_threshold or y >= self.collision_threshold or y <= -self.collision_threshold:
            cost['cost_ltl_walls'] = 1
        return cost

    def wall_sensor(self, x, y):
        return np.array([
            self.calculate_wall_distance(pos, threshold)
            for pos, threshold in
            [(x, self.collision_threshold), (y, -self.collision_threshold), (x, -self.collision_threshold),
             (y, self.collision_threshold)]
        ])

    @staticmethod
    def calculate_wall_distance(pos: float, threshold: float, gain: float = 1):
        return np.exp(-gain * np.abs(pos - threshold))

    @property
    def pos(self):
        """Helper to get list of Sigwalls positions."""
