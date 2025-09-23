# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Point-mass domain."""

import collections

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.rl.control import _spec_from_observation

from ltl.logic import FrozenAssignment, Assignment

_DEFAULT_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    with open(f'src/envs/dmc/ltl_point_mass/point_mass.xml', mode='rb') as f:
        xml_string = f.read()
    return xml_string, common.ASSETS


@SUITE.add('ltl')
def ltl(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the LTL task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = LTLPointMass(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
    """Physics for the point_mass domain."""

    def point_mass_position(self):
        return self.named.data.geom_xpos['pointmass']


class LTLPointMass(base.Task):
    """The PointMass LTL task"""

    def __init__(self, random=None):
        """Initialize an instance of `PointMass`.

        Args:
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        super().__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Args:
          physics: An instance of `mujoco.Physics`.
        """
        self.randomize_position(physics, self.random)
        super().initialize_episode(physics)

    @staticmethod
    def randomize_position(physics, random):
        for joint_name in ['root_x', 'root_y']:
            physics.named.data.qpos[joint_name] = random.uniform(-.25, .25)

    def get_observation(self, physics):
        """Returns an observation of the state."""
        obs = collections.OrderedDict()
        obs['position'] = physics.position()
        obs['velocity'] = physics.velocity()
        obs['propositions'] = self.compute_active_propositions(physics)
        obs['terminated'] = False
        return obs

    def get_reward(self, physics):
        return 0

    def observation_spec(self, physics):
        # This is a hack, since the label and terminated information is not actually part of the observation
        observation = self.get_observation(physics)
        del observation['propositions']
        del observation['terminated']
        return _spec_from_observation(observation)

    @staticmethod
    def compute_active_propositions(physics):
        pos = physics.point_mass_position()
        x, y = pos[0], pos[1]
        if .1 <= x <= .2 and .1 <= y <= .2:
            return {'green'}
        elif -.2 <= x <= -.1 and .1 <= y <= .2:
            return {'blue'}
        elif -.2 <= x <= -.1 and -.2 <= y <= -.1:
            return {'red'}
        elif .1 <= x <= .2 and -.2 <= y <= -.1:
            return {'yellow'}
        return set()

    @staticmethod
    def get_propositions() -> list[str]:
        return sorted(['green', 'blue', 'red', 'yellow'])

    def get_impossible_assignments(self) -> set[FrozenAssignment]:
        return Assignment.more_than_one_true_proposition(set(self.get_propositions()))
