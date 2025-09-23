from .flatworld import FlatWorld

from gymnasium.envs.registration import register

register(
    id='FlatWorld-v0',
    entry_point='envs.flatworld.flatworld:FlatWorld',
    kwargs=dict(
        continuous_actions=False
    )
)

register(
    id='FlatWorldSafety-v0',
    entry_point='envs.flatworld.flatworld:FlatWorldSafety',
    kwargs=dict(
        continuous_actions=False
    )
)