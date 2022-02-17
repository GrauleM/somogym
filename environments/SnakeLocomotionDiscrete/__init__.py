import sys

from gym.envs.registration import register

default_max_episode_steps = 100

register(
    id="SnakeLocomotionDiscrete-v0",
    entry_point="environments.SnakeLocomotionDiscrete.SnakeLocomotionDiscrete:SnakeLocomotionDiscrete",
    max_episode_steps=default_max_episode_steps,
)
