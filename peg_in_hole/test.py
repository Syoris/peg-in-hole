env = YourEnv()
obs, info = env.reset()
n_steps = 10
for _ in range(n_steps):
    # Random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if done:
        obs, info = env.reset()


def test(cfg):
    logger.info('Testing trained model')

    """ Test the trained model """
    vec_env = model.get_env()
    vec_env.render_mode = 'human'

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render('human')
        ...
