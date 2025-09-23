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

import numpy as np
from safety_gymnasium.assets.group import GROUP
from safety_gymnasium.bases.base_object import Geom


class Zones(Geom):  # pylint: disable=too-many-instance-attributes
    """Colored zones."""

    COLORS = {
        "black": np.array([0, 0, 0, 1]),
        "blue": np.array([0, 0, 1, 1]),
        "green": np.array([0, 1, 0, 1]),
        "cyan": np.array([0, 1, 1, 1]),
        "red": np.array([1, 0, 0, 1]),
        "magenta": np.array([1, 0, 1, 1]),
        "yellow": np.array([1, 1, 0, 1]),
        "white": np.array([1, 1, 1, 1]),
        "orange": np.array([1.0, 0.5, 0.0, 1.0]),
        "purple": np.array([0.5, 0.0, 0.5, 1.0]),
        "lime": np.array([0.5, 1.0, 0.0, 1.0]),
        "teal": np.array([0.0, 0.5, 0.5, 1.0]),
        "pink": np.array([1.0, 0.75, 0.8, 1.0]),
        "brown": np.array([0.6, 0.3, 0.0, 1.0]),
        "navy": np.array([0.0, 0.0, 0.5, 1.0]),
        "gray": np.array([0.5, 0.5, 0.5, 1.0]),
    }

    def __init__(self, color: str, size: float, num: int, locations=None, keepout=0.55):
        self.color_name = color
        self.name = f'{color}_zones'
        self.num = num
        self.size: float = size
        self.placements: list = None  # Placements list for hazards (defaults to full extents)
        self.locations: list = locations if locations else []  # Fixed locations to override placements
        self.keepout: float = keepout  # Radius of hazard keepout for placement
        self.alpha: float = 0.25
        
        if self.color_name not in self.COLORS:
            self.color = self.COLORS['black']
        else:
            self.color: np.array = self.COLORS[self.color_name]
        self.group: int = self.calculate_group()
        self.is_lidar_observed: bool = True
        self.is_constrained: bool = True

    def calculate_group(self) -> int:
        max_predefined_group = max(GROUP.values())
        return max_predefined_group + sorted(self.COLORS.keys()).index(self.color_name) + 1

    def get_config(self, xy_pos, rot):
        """To facilitate get specific config for this object."""
        body = {
            'name': self.name,
            'pos': np.r_[xy_pos, 2e-2],  # self.hazards_size / 2 + 1e-2],
            'rot': rot,
            'geoms': [
                {
                    'name': self.name,
                    'size': [self.size, 1e-2],  # self.hazards_size / 2],
                    'type': 'cylinder',
                    'contype': 0,
                    'conaffinity': 0,
                    'group': self.group,
                    'rgba': self.color * np.array([1.0, 1.0, 1.0, self.alpha]),
                },
            ],
        }
        return body

    def cal_cost(self):
        cost = {f'cost_zones_{self.color_name}': 0}
        for h_pos in self.pos:
            h_dist = self.agent.dist_xy(h_pos)
            if h_dist <= self.size:
                cost[f'cost_zones_{self.color_name}'] = 1
        return cost

    @property
    def pos(self):
        """Helper to get the hazards positions from layout."""
        # pylint: disable-next=no-member
        return [self.engine.data.body(f'{self.name[:-1]}{i}').xpos.copy() for i in range(self.num)]
