from gymnasium.envs.registration import register

register(
    id='vxUnjamming-v0',
    entry_point='peg_in_hole.tasks.kinova_gen2_env:KinovaGen2Env',
    max_episode_steps=300,
)