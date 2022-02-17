from gym.envs.registration import register

default_max_episode_steps = 100

register(
    id="AntipodalGripper-v0",
    entry_point="environments.AntipodalGripper.AntipodalGripper:AntipodalGripper",
    max_episode_steps=default_max_episode_steps,
)
