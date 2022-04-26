"""
tests to ensure that two distinct environments can be run from the same script

also does some open and closing of the envs to make sure all of that is handled appropriately
"""

import yaml
import gym
from pathlib import Path
import os
import sys

import pytest

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)
from environments.utils.import_handler import import_environment


def load_env(
    env_name, render=False
):  # todo: this should probably be a utility for general use

    # get the default run config path
    run_config_file = (
        Path(os.path.dirname(__file__))
        / "../environments"
        / env_name
        / "benchmark_run_config.yaml"
    )

    with open(run_config_file, "r") as config_file:
        run_config = yaml.safe_load(config_file)

    debug = False

    # prepare env
    import_environment(env_name)
    env = gym.make(
        run_config["env_id"],
        run_config=run_config,
        run_ID=f"{env_name}-sim_test",
        # todo: may be better to check whether the run ID exists already
        render=render,
        debug=debug,
    )
    return env


def test_simultaneous_runs_different_envs():
    """
    tests simultaneous running of two environments with different names; both without rendering
    """

    env1 = load_env("PlanarBlockPushing", render=False)
    env2 = load_env("PlanarReaching")

    env1.reset()
    env2.reset()

    n_steps = 10
    for i in range(n_steps):
        env1.step(env1.action_space.sample())
        env2.step(env2.action_space.sample())

    # make sure the envs can be reset and run again
    env1.reset()
    env2.reset()
    for i in range(n_steps):
        env1.step(env1.action_space.sample())
        env2.step(env2.action_space.sample())

    env1.close()
    env2.close()


@pytest.mark.gui
def test_simultaneous_runs_different_envs_gui():
    """
    tests simultaneous running of two environments with different names; both without rendering
    """

    env1 = load_env("PlanarBlockPushing", render=True)
    env2 = load_env("PlanarReaching")

    env1.reset()
    env2.reset()

    n_steps = 10
    for i in range(n_steps):
        env1.step(env1.action_space.sample())
        env2.step(env2.action_space.sample())

    env1.close()
    env2.close()


def test_simultaneous_runs_identical_envs():
    """
    tests simultaneous running of two environments with different names; both without rendering
    """

    env1 = load_env("PlanarBlockPushing", render=False)
    env2 = load_env("PlanarBlockPushing")

    env1.reset()
    env2.reset()

    n_steps = 10
    for i in range(n_steps):
        env1.step(env1.action_space.sample())
        env2.step(env2.action_space.sample())

    # make sure the envs can be reset and run again
    env1.reset()
    env2.reset()
    for i in range(n_steps):
        env1.step(env1.action_space.sample())
        env2.step(env2.action_space.sample())

    # make sure one env can be closed and the other one can be reset still and run again
    env1.close()
    env2.reset()
    for i in range(n_steps):
        env2.step(env2.action_space.sample())

    env2.close()


if __name__ == "__main__":
    test_simultaneous_runs_different_envs()
    test_simultaneous_runs_identical_envs()
