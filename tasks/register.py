from gymnasium.envs.registration import register

register(
    id='Laikago-v0',
    entry_point='tasks.laikago.env_v0:Env',
    max_episode_steps=1000,
)

register(
    id='Cassie-v0',
    entry_point='tasks.cassie.env_v0:Env',
    max_episode_steps=1000,
)
