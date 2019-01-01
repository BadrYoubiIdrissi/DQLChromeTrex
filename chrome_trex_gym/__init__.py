from gym.envs.registration import register

register(
    id='chrome-trex-v0',
    entry_point='chrome_trex_gym.envs:ChromeTrexEnv',
    kwargs={'FPS': 60, 'headless': False}
)

register(
    id='chrome-trex-train-v0',
    entry_point='chrome_trex_gym.envs:ChromeTrexEnv',
    kwargs={'FPS': 0, 'headless': True}
)

register(
    id='chrome-trex-train-render-v0',
    entry_point='chrome_trex_gym.envs:ChromeTrexEnv',
    kwargs={'FPS': 0, 'headless': False}
)