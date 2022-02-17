from gym.envs.registration import register

default_max_episode_steps = 100

register(
    id="PenSpinner-v0",
    entry_point="environments.PenSpinner.PenSpinner:PenSpinner",
    max_episode_steps=default_max_episode_steps,
)
