import os
import sys

import pytest

import argparse
from copy import deepcopy
import numpy as np
import pybullet as p
import sorotraj
import yaml
from pathlib import Path

import gym
from stable_baselines3.common.utils import set_random_seed

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from environments.utils.import_handler import import_environment


def somogym_step_tester(
    env_name,
    render=False,
    debug=False,
    total_env_steps=5,
):
    run_config_file = (
        Path(os.path.dirname(__file__))
        / "../environments"
        / env_name
        / "benchmark_run_config.yaml"
    )

    with open(run_config_file, "r") as config_file:
        run_config = yaml.safe_load(config_file)

    # prepare env
    import_environment(env_name)
    env = gym.make(
        run_config["env_id"],
        run_config=run_config,
        run_ID=f"{env_name}-step_test",
        render=render,
        debug=debug,
    )

    run_config["seed"] = 10110
    set_random_seed(run_config["seed"])
    env.seed(run_config["seed"])
    env.reset()

    # run env for total_env_steps steps
    for _ in range(total_env_steps):
        env.step(env.action_space.sample())  # take a random action

    # make sure seeding works correctly for this env
    # seed once, reset, and take a step
    set_random_seed(run_config["seed"])
    env.seed(run_config["seed"])
    env.reset()
    action_a = env.action_space.sample()
    step_result_a = env.step(action_a)  # take a random action
    # seed and reset again and take another step
    set_random_seed(run_config["seed"])
    env.seed(run_config["seed"])
    env.reset()
    action_b = env.action_space.sample()
    step_result_b = env.step(action_b)  # take a random action

    # compare results
    assert (
        step_result_a[0] == step_result_b[0]
    ).all(), f"seeding does not work correctly for env {env_name}: observations are inconsistent"
    assert (
        step_result_a[1] == step_result_b[1]
    ), f"seeding does not work correctly for env {env_name}: rewards are inconsistent"
    assert (
        step_result_a[2] == step_result_b[2]
    ), f"seeding does not work correctly for env {env_name}: done flags are inconsistent"
    assert (
        step_result_a[3] == step_result_b[3]
    ), f"seeding does not work correctly for env {env_name}: info entries are inconsistent"

    # finally, close the env
    env.close()


# ANTIPODAL GRIPPER
def test_AntipodalGripper_step():
    somogym_step_tester("AntipodalGripper")


@pytest.mark.gui
def test_AntipodalGripper_step_gui():
    somogym_step_tester("AntipodalGripper", render=True, total_env_steps=100)


# IN-HAND MANIPULATION
def test_InHandManipulation_step():
    somogym_step_tester("InHandManipulation")


@pytest.mark.gui
def test_InHandManipulation_step_gui():
    somogym_step_tester("InHandManipulation", render=True, total_env_steps=100)


# IN-HAND MANIPULATION INVERTED
def test_InHandManipulationInverted_step():
    somogym_step_tester("InHandManipulationInverted")


@pytest.mark.gui
def test_InHandManipulationInverted_step_gui():
    somogym_step_tester("InHandManipulationInverted", render=True, total_env_steps=100)


# PEN SPINNER
def test_PenSpinner_step():
    somogym_step_tester("PenSpinner")


@pytest.mark.gui
def test_PenSpinner_step_gui():
    somogym_step_tester("PenSpinner", render=True, total_env_steps=100)


# PLANAR BLOCK PUSHING
def test_PlanarBlockPushing_step():
    somogym_step_tester("PlanarBlockPushing")


@pytest.mark.gui
def test_PlanarBlockPushing_step_gui():
    somogym_step_tester("PlanarBlockPushing", render=True, total_env_steps=100)


# PLANAR REACHING
def test_PlanarReaching_step():
    somogym_step_tester("PlanarReaching")


@pytest.mark.gui
def test_PlanarReaching_step_gui():
    somogym_step_tester("PlanarReaching", render=True, total_env_steps=100)


# PLANAR REACHING OBSTACLE
def test_PlanarReachingObstacle_step():
    somogym_step_tester("PlanarReachingObstacle")


@pytest.mark.gui
def test_PlanarReachingObstacle_step_gui():
    somogym_step_tester("PlanarReachingObstacle", render=True, total_env_steps=100)


# SNAKE LOCOMOTION DISCRETE
def test_SnakeLocomotionDiscrete_step():
    somogym_step_tester("SnakeLocomotionDiscrete")


@pytest.mark.gui
def test_SnakeLocomotionDiscrete_step_gui():
    somogym_step_tester("SnakeLocomotionDiscrete", render=True, total_env_steps=100)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gui":
        test_AntipodalGripper_step_gui()
        test_InHandManipulation_step_gui()
        test_InHandManipulationInverted_step_gui()
        test_PenSpinner_step_gui()
        test_PlanarBlockPushing_step_gui()
        test_PlanarReaching_step_gui()
        test_PlanarReachingObstacle_step_gui()
        test_SnakeLocomotionDiscrete_step_gui()
    else:
        test_AntipodalGripper_step()
        test_InHandManipulation_step()
        test_InHandManipulationInverted_step()
        test_PenSpinner_step()
        test_PlanarBlockPushing_step()
        test_PlanarReaching_step()
        test_PlanarReachingObstacle_step()
        test_SnakeLocomotionDiscrete_step()
