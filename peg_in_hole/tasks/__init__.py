from gymnasium.envs.registration import register

register(
    id='vxUnjamming-v0',
    entry_point='peg_in_hole.tasks.RPL_Insert_3DoF:RPL_Insert_3DoF',
    max_episode_steps=300,
)
