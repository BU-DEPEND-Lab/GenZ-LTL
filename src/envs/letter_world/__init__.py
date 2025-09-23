from gymnasium.envs.registration import register

register(
    id='LetterEnv-v0',
    entry_point='envs.letter_world.letter_env:LetterEnv',
    kwargs=dict(
        grid_size=7,
        letters="aabbccddeeffgghhiijjkkll",
        use_fixed_map=False,
        use_agent_centric_view=True,
    )
)

register(
    id='LetterSafetyEnv-v0',
    entry_point='envs.letter_world.letter_env:LetterSafetyEnv',
    kwargs=dict(
        grid_size=7,
        letters="aabbccddeeffgghhiijjkkll",
        use_fixed_map=False,
        use_agent_centric_view=True,
    )
)

register(
    id='LetterEnv-v0.fixed',
    entry_point='envs.letter_world.letter_env:LetterEnv',
    kwargs=dict(
        grid_size=7,
        letters="abcdefghijkl",
        use_fixed_map=True,
        use_agent_centric_view=True,
        map={
            (4, 0): 'c',
            (0, 2): 'a',
            (0, 1): 'd',
            (0, 3): 'd',
            (1, 2): 'd',
            (6, 2): 'd',
            (3, 4): 'a'
        }
    )
)


register(
    id='LetterEnv-v0.multi_avoid',
    entry_point='envs.letter_world.letter_env:LetterEnv',
    kwargs=dict(
        grid_size=7,
        letters="abcdefghijkl",
        use_fixed_map=True,
        use_agent_centric_view=True,
        map={
            (3, 1): 'a',
            (1, 3): 'b',
            (5, 3): 'a',
            (2, 2): 'b',
            (3, 5): 'b',
            (3, 3): 'c'
        }
    )
)
