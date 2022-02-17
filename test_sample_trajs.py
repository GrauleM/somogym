import os, sys
import pytest
import yaml

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from pathlib import Path
from sample_trajectories.run_traj import gen_expert_data as run_traj


def somogym_sample_traj_tester(env_name, traj_name, render=False, debug=False):
    run_config_file = (
        Path(os.path.dirname(__file__))
        / "../environments"
        / env_name
        / "benchmark_run_config.yaml"
    )
    with open(run_config_file, "r") as config_file:
        run_config = yaml.safe_load(config_file)
    run_config["expert_name"] = traj_name
    run_traj(
        env_name,
        run_config,
        num_steps=20,
        run_render=render,
        debug=debug,
        record_data=False,
    )


# ANTIPODAL GRIPPER
def test_AntipodalGripper_sample_traj():
    somogym_sample_traj_tester("AntipodalGripper", "AntipodalGripper-grip_lift")


@pytest.mark.gui
def test_AntipodalGripper_sample_traj_gui():
    somogym_sample_traj_tester(
        "AntipodalGripper", "AntipodalGripper-grip_lift", render=True
    )


# IN-HAND MANIPULATION
def test_InHandManipulation_sample_traj():
    somogym_sample_traj_tester("InHandManipulation", "InHandManipulation-gaiting")


@pytest.mark.gui
def test_InHandManipulation_sample_traj_gui():
    somogym_sample_traj_tester(
        "InHandManipulation", "InHandManipulation-gaiting", render=True
    )


# IN-HAND MANIPULATION INVERTED
def test_InHandManipulationInverted_sample_traj():
    somogym_sample_traj_tester(
        "InHandManipulationInverted", "InHandManipulationInverted-gaiting"
    )


@pytest.mark.gui
def test_InHandManipulationInverted_sample_traj_gui():
    somogym_sample_traj_tester(
        "InHandManipulationInverted", "InHandManipulationInverted-gaiting", render=True
    )


# PEN SPINNER
def test_PenSpinner_sample_traj():
    somogym_sample_traj_tester("PenSpinner", "PenSpinner-twist")


@pytest.mark.gui
def test_PenSpinner_sample_traj_gui():
    somogym_sample_traj_tester("PenSpinner", "PenSpinner-twist", render=True)


# PLANAR BLOCK PUSHING
def test_PlanarBlockPushing_sample_traj():
    somogym_sample_traj_tester("PlanarBlockPushing", "PlanarBlockPushing-grasp")


@pytest.mark.gui
def test_PlanarBlockPushing_sample_traj_gui():
    somogym_sample_traj_tester(
        "PlanarBlockPushing", "PlanarBlockPushing-grasp", render=True
    )


# PLANAR REACHING
def test_PlanarReaching_sample_traj():
    somogym_sample_traj_tester("PlanarReaching", "PlanarReaching-reach")


@pytest.mark.gui
def test_PlanarReaching_sample_traj_gui():
    somogym_sample_traj_tester("PlanarReaching", "PlanarReaching-reach", render=True)


# PLANAR REACHING OBSTACLE
def test_PlanarReachingObstacle_sample_traj():
    somogym_sample_traj_tester("PlanarReachingObstacle", "PlanarReachingObstacle-lean")


@pytest.mark.gui
def test_PlanarReachingObstacle_sample_traj_gui():
    somogym_sample_traj_tester(
        "PlanarReachingObstacle", "PlanarReachingObstacle-lean", render=True
    )


# SNAKE LOCOMOTION DISCRETE (aka LocoSnake)
def test_SnakeLocomotionDiscrete_sample_traj():
    somogym_sample_traj_tester(
        "SnakeLocomotionDiscrete", "SnakeLocomotionDiscrete-slither"
    )


@pytest.mark.gui
def test_SnakeLocomotionDiscrete_sample_traj_gui():
    somogym_sample_traj_tester(
        "SnakeLocomotionDiscrete", "SnakeLocomotionDiscrete-slither", render=True
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gui":
        test_AntipodalGripper_sample_traj_gui()
        test_InHandManipulation_sample_traj_gui()
        test_InHandManipulationInverted_sample_traj_gui()
        test_PenSpinner_sample_traj_gui()
        test_PlanarBlockPushing_sample_traj_gui()
        test_PlanarReaching_sample_traj_gui()
        test_PlanarReachingObstacle_sample_traj_gui()
        test_SnakeLocomotionDiscrete_sample_traj_gui()
    else:
        test_AntipodalGripper_sample_traj()
        test_InHandManipulation_sample_traj()
        test_InHandManipulationInverted_sample_traj()
        test_PenSpinner_sample_traj()
        test_PlanarBlockPushing_sample_traj()
        test_PlanarReaching_sample_traj()
        test_PlanarReachingObstacle_sample_traj()
        test_SnakeLocomotionDiscrete_sample_traj()
        test_InchwormLocomotion_sample_traj()
