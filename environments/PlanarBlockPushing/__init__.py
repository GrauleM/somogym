import sys

from gym.envs.registration import register

default_max_episode_steps = 100

register(
    id="PlanarBlockPushing-v0",
    entry_point="environments.PlanarBlockPushing.PlanarBlockPushing:PlanarBlockPushing",
    max_episode_steps=default_max_episode_steps,
)
