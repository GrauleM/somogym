import os
import sys

import gym
import argparse
from copy import deepcopy
import numpy as np
import pybullet as p
import sorotraj
import yaml
from pathlib import Path

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from environments.utils import import_handler

# expert directory should be of the following form. vid.mp4 and data directory w/ contents will be automatically generated.
# ../experts/
#       expert_1/
#           example_traj.yaml
#           vid.mp4
#           data/
#               curve.npy
#               position.npy
#               torque.npy
#               velocity.npy
#       expert_2/
#           another_traj.yaml


def gen_expert_data(
    environment_name,
    run_config,
    num_steps=1000,
    run_render=True,
    debug=False,
    record_data=True,
):

    import_handler.import_environment(environment_name)

    env = gym.make(
        run_config["env_id"],
        run_config=run_config,
        run_ID=str(run_config["expert_name"]),
        render=run_render,
        debug=debug,
    )

    # todo: only seed when required, make seed selectable; this should use the seed from the run_config
    env.seed(0)

    if "max_episode_steps" not in run_config:
        run_config["max_episode_steps"] = num_steps
    num_steps = run_config["max_episode_steps"]

    expert_dir_abs_path = Path(os.path.dirname(__file__))
    if "expert_dir_abs_path" in run_config:
        expert_dir_abs_path = run_config["expert_dir_abs_path"]
    expert_abs_path = Path(expert_dir_abs_path) / run_config["expert_name"]
    expert_rel_path = Path(os.path.relpath(expert_abs_path))
    expert_data_dir = expert_rel_path / "data"
    expert_traj_def_file = expert_rel_path / "traj.yaml"
    if record_data:
        os.makedirs(expert_data_dir, exist_ok=True)

    vid_path = expert_rel_path / "vid.mp4"
    obs = env.reset(run_render=run_render)

    # Define a conversion function from real trajectories (differential pressure) to somo actuation torques.
    # todo: get rid of all somo traj dependencies...
    def real2somo(traj_line, weights):
        traj_length = len(traj_line) - 1
        traj_line_new = [0] * (traj_length + 1)
        traj_line_new[0] = traj_line[0]  # Use the same time point
        for idx in range(int(traj_length / 2)):
            idx_list = [2 * idx + 1, 2 * idx + 2]
            traj_line_new[idx_list[0]] = (
                weights[0] * (traj_line[idx_list[0]] + traj_line[idx_list[1]]) / 2.0
            )
            traj_line_new[idx_list[1]] = (
                weights[1] * (traj_line[idx_list[0]] - traj_line[idx_list[1]]) / 2.0
            )
        if traj_length % 2 != 0:
            traj_line_new[-1] = traj_line[-1]
        return traj_line_new

    # use like this
    weights = [2.0, -6.5]
    conversion_real2somo = lambda line: real2somo(line, weights)
    traj_build = sorotraj.TrajBuilder(graph=False)
    traj_build.load_traj_def(str(expert_traj_def_file))
    # traj_build.convert(conversion_real2somo)
    # Todo: remove all holdovers from real2somo and make a standalone system & example for mapping trajectories to hardware
    # and to run hardware differential pressure trajectories in simulation.
    # afterwards delete all unused ugly code.

    trajectory = traj_build.get_trajectory()
    interp = sorotraj.Interpolator(trajectory)
    action_len = env.action_space.shape[0]

    def traj_actuation_fn(time):
        if time == 0:
            return [0] * action_len
        traj_interp_fn = interp.get_interp_function(
            num_reps=30,  # change num repos here
            speed_factor=1.0,
            invert_direction=True,  # todo: this is hacky - change expert traj entries and set invert_direction=False
            as_list=False,
        )
        return traj_interp_fn(time)

    torques = [None] * num_steps
    positions = [None] * num_steps
    # angles = [None] * num_steps
    # curves = [None] * num_steps
    velocities = [None] * num_steps

    if run_render and record_data:
        if vid_path.exists():
            os.remove(vid_path)
        vid_filename = os.path.abspath(vid_path)
        logIDvideo = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, vid_filename)

    for i in range(num_steps):
        applied_action = (
            np.array(traj_actuation_fn(env.step_count * env.action_time))
            / env.torque_multiplier
        )
        restricted_action = np.minimum(
            np.maximum(applied_action, np.array([-1.0] * action_len)),
            np.array([1.0] * action_len),
        )
        _obs, _rewards, _dones, info = env.step(restricted_action)

        # TODO: seperate record_data functionality into its own file.
        if record_data:
            manipulator_states = env.get_manipulator_states(
                components=("positions", "velocities")
            )
            positions_at_step = [None] * env.n_manipulators
            velocities_at_step = [None] * env.n_manipulators

            for m, m_state in enumerate(manipulator_states):
                positions_at_step[m] = deepcopy(m_state["positions"])
                velocities_at_step[m] = deepcopy(m_state["velocities"])

            torques[i] = deepcopy(env.applied_torque.flatten())
            positions[i] = deepcopy(np.array(positions_at_step).flatten())
            velocities[i] = deepcopy(np.array(velocities_at_step).flatten())

        if run_render:
            env.render()

    if run_render and record_data:
        p.stopStateLogging(logIDvideo)

    env.close()

    if record_data:
        np.save(expert_data_dir / "torques.npy", np.array(torques))
        np.save(expert_data_dir / "positions.npy", np.array(positions))
        np.save(expert_data_dir / "velocities.npy", np.array(velocities))


def main():
    parser = argparse.ArgumentParser(
        description="Arguments to run an expert trajectory and optionally record data."
    )
    parser.add_argument("-env", "--env_name", help="Environment name.", required=True)
    parser.add_argument(
        "-t",
        "--traj_name",
        help="Trajectory name.",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--num_steps",
        help="Number of steps to run.",
        required=False,
        default=1000,
    )
    parser.add_argument(
        "-v",
        "--render",
        help="Render the environment.",
        action="store_true",
    )
    parser.add_argument(
        "-d", "--debug", help="Display SoMo-RL Debugging Dashboard", action="store_true"
    )
    parser.add_argument(
        "-dl",
        "--debug_list",
        nargs="+",
        help="List of debugger components to show in panel (space separated). Choose from reward_components, observations, actions, applied_torques",
        required=False,
        default=[],
    )
    parser.add_argument(
        "-s",
        "--save",
        help="Save the expert as data is recorded.",
        action="store_true",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        help="Run config absolute path.",
        required=False,
        default=None,
    )

    arg = parser.parse_args()

    if arg.config_path is not None:
        run_config_file = arg.config_path
    else:
        run_config_file = (
            Path(os.path.dirname(__file__))
            / "../environments"
            / arg.env_name
            / "benchmark_run_config.yaml"
        )

    with open(run_config_file, "r") as config_file:
        run_config = yaml.safe_load(config_file)

    run_config["expert_name"] = arg.traj_name

    debug = arg.debug
    if len(arg.debug_list) > 0:
        debug = deepcopy(arg.debug_list)

    gen_expert_data(
        arg.env_name,
        run_config,
        num_steps=int(arg.num_steps),
        run_render=arg.render,
        debug=debug,
        record_data=arg.save,
    )  # todo: rename gen_expert_data function?


if __name__ == "__main__":
    main()
