import numpy as np
import os
import pybullet as p

from somo.sm_manipulator_definition import SMManipulatorDefinition
from somo.sm_continuum_manipulator import SMContinuumManipulator

from environments import SomoEnv
from environments.PlanarReaching.env_utils import (
    target_start_funcs,
    obstacle_start_funcs,
)


class PlanarReaching(SomoEnv.SomoEnv):
    def __init__(
        self,
        run_config=None,
        run_ID=None,
        render=False,
        debug=0,
        with_obstacle=False,
    ):

        # load the manipulator definition
        manipulator_def_file = os.path.join(
            os.path.dirname(__file__), "definitions/manipulator_definition.yaml"
        )
        manipulator_def = SMManipulatorDefinition.from_file(manipulator_def_file)
        manipulator_def.urdf_filename = os.path.join(
            os.path.dirname(__file__),
            "definitions/" + manipulator_def.manipulator_name + ".urdf",
        )

        active_actuators_list = [([0, 1, 2, 3], [0, 0, 0, 0])]
        manipulators = [SMContinuumManipulator(manipulator_def)]

        start_positions = [[0, 0, 0]]
        start_orientations = [p.getQuaternionFromEuler([0, 0, 0])]
        manipulator_base_constraints = ["static"]

        action_shape = (4,)

        # set some geometry params
        self.target_start_pos = [0.5] + run_config["target_start_pos"][1:]
        self.target_start_or = p.getQuaternionFromEuler(run_config["target_start_or"])

        # this is where a randomized sampling fn could be provided to change the target start at random
        self.get_target_start = None
        if "target_start_func" in run_config:
            self.get_target_start = target_start_funcs.target_start_funcs[
                run_config["target_start_func"]
            ]

        try:
            self.with_obstacle = with_obstacle or run_config["with_obstacle"]
        except:
            self.with_obstacle = with_obstacle

        self.get_obstacle_start = None
        if self.with_obstacle:
            self.obstacle_scale = run_config["obstacle_scale"]
            self.obstacle_pos = [0, 1.5, 6.0]
            self.obstacle_or = p.getQuaternionFromEuler([0, 0, 0])
            if "obstacle_start_func" in run_config:
                self.get_obstacle_start = obstacle_start_funcs.obstacle_start_funcs[
                    run_config["obstacle_start_func"]
                ]

        self.gravity_factor = run_config["gravity_factor"]

        if "action_noise" in run_config:
            self.action_noise = run_config["action_noise"]
            if (
                self.action_noise
            ):  # only do the assertions below if action_noise is not None
                assert (
                    len(self.action_noise) == 2
                ), f"action_noise has to be a list of lists with means and stand deviations"
                assert len(self.action_noise[0]) == len(
                    self.action_noise[1]
                ), f"action_noise means and stds have to be the same dimension"
                assert (
                    len(self.action_noise[0]),
                ) == action_shape, (
                    f"action_noise means and stds have to be the same as action_shape"
                )
        else:
            self.action_noise = None

        super().__init__(
            manipulators=manipulators,
            start_positions=start_positions,
            start_orientations=start_orientations,
            manipulator_base_constraints=manipulator_base_constraints,
            active_actuators_list=active_actuators_list,
            action_shape=action_shape,
            run_config=run_config,
            gravity=[
                0,
                0,
                196.2 * self.gravity_factor,
            ],  # scale factor 20 applied to gravity, then divided it by 4 to make the task easier
            ground_plane_height=None,
            run_ID=run_ID,
            render=render,
            debug=debug,
        )

    def step(self, action):

        if (
            self.action_noise
        ):  # todo: move this deeper into somoenv; make sure that action limit / action space is handled correctly
            means = self.action_noise[0]
            stds = self.action_noise[1]
            action += self.np_random.normal(means, stds)

            for i in range(self.action_space.shape[0]):
                if (
                    action[i] > self.action_space.high[i]
                ):  # assert action limits are obeyed
                    action[i] = self.action_space.high[i]
                elif action[i] < self.action_space.low[i]:
                    action[i] = self.action_space.low[i]

        obs, reward, done, info = super().step(action=action)

        try:
            failure_penalty = self.run_config["failure_penalty"]
        except:
            failure_penalty = 0

        if done:
            reward -= failure_penalty
        return (obs, reward, done, info)

    def get_observation_shape(self):
        # possible observation flags: target_pos, target_or, torques
        obs_dimension_per_sample = {
            "target_pos": 2,
            "target_or": 1,
            "target_velocity": 2,
            "positions": 2,
            "velocities": 2,
            "tip_pos": 2,
            "tip_target_dist_vector": 2,
            "angles": 1,
            "curvatures": 1,
            "applied_input_torques": 4,
            "tip_orientation_error": 1,
        }

        obs_flags = self.run_config["observation_flags"]
        obs_len = 0
        for f in obs_flags:
            num_pts = obs_flags[f] if obs_flags[f] is not None else 1
            obs_len += num_pts * obs_dimension_per_sample[f]
        return (obs_len,)

    def get_observation(self):
        obs_flags = self.run_config["observation_flags"]

        self.manipulator_states = self.get_manipulator_states(
            components=list(obs_flags.keys())
        )
        state = np.array([])

        # todo: add contact observations

        # TODO MAJOR: update observations. they were written for manipulator in xy plane, but now we are in a different plane
        # todo: related major todo: double check this in all envs
        if "target_pos" in obs_flags or "target_or" in obs_flags:
            target_pos, target_or_quat = p.getBasePositionAndOrientation(self.target_id)
            if "target_pos" in obs_flags:
                state = np.concatenate((state, np.array(target_pos)[1:]))
            if "target_or" in obs_flags:
                target_or = p.getEulerFromQuaternion(target_or_quat)
                state = np.concatenate((state, np.array([target_or[0]])))

        if "target_velocity" in obs_flags:
            target_velocity = np.array(
                p.getBaseVelocity(bodyUniqueId=self.target_id)[0]
            )
            state = np.concatenate((state, np.array(target_velocity)[1:]))

        if "positions" in obs_flags:
            positions = np.array(
                [state["positions"][:, 1:] for state in self.manipulator_states]
            )
            if obs_flags["positions"]:
                positions = np.array(
                    [
                        self.reduce_state_len(ps, obs_flags["positions"])
                        for ps in positions
                    ]
                )
            state = np.concatenate((state, positions.flatten()))

        if "velocities" in obs_flags:
            velocities = np.array(
                [state["velocities"][:, 1:] for state in self.manipulator_states]
            )
            if obs_flags["velocities"]:
                velocities = np.array(
                    [
                        self.reduce_state_len(vs, obs_flags["velocities"])
                        for vs in velocities
                    ]
                )
            state = np.concatenate((state, velocities.flatten()))

        if "tip_pos" in obs_flags or "tip_target_dist_vector" in obs_flags:
            positions = np.array(
                [state["positions"][:, 1:] for state in self.manipulator_states]
            )
            tip_pos = positions[0][-1]
            if "tip_target_dist_vector" in obs_flags:
                target_pos, _ = p.getBasePositionAndOrientation(self.target_id)
                state = np.concatenate((state, target_pos[1:] - tip_pos))
            if "tip_pos" in obs_flags:
                state = np.concatenate((state, tip_pos))

        if "angles" in obs_flags:
            # todo: this backbone angle stuff seems to have a problem... rethink how to read this - maybe get joint angles instead. try to get to angles that are 0 in one of the axes in case of planar manipulators
            angles = np.array([state["angles"] for state in self.manipulator_states])
            if obs_flags["angles"]:
                angles = np.array(
                    [
                        self.reduce_state_len(angs, obs_flags["angles"])
                        for angs in angles
                    ]
                )
            state = np.concatenate((state, angles.flatten()))

        if "curvatures" in obs_flags:
            curvatures = np.array(
                [state["curvatures"] for state in self.manipulator_states]
            )
            if obs_flags["curvatures"]:
                curvatures = np.array(
                    [
                        self.reduce_state_len(cs, obs_flags["curvatures"])
                        for cs in curvatures
                    ]
                )
            state = np.concatenate((state, curvatures.flatten()))

        if "applied_input_torques" in obs_flags:
            applied_input_torques = np.array(self.applied_torque)
            state = np.concatenate((state, applied_input_torques.flatten()))

        if "tip_orientation_error" in obs_flags:
            assert self.target_x_orientation, (
                f"you are requesting to observe tip orientation error, but "
                f"self.target_x_orientation has not been initialized. double check the "
                f"target function you use to assign target position and orientation."
            )

            num_links = len(self.manipulators[0].flexible_joint_indices)
            last_link_state = p.getLinkState(
                self.manipulators[0].bodyUniqueId, num_links - 1
            )
            tip_orientation_x = p.getEulerFromQuaternion(last_link_state[1])[0]
            orientation_error = self.target_x_orientation - tip_orientation_x

            # take care of angle wrapping
            if -np.pi < orientation_error < np.pi:
                orientation_error_wrapped = orientation_error
            elif np.pi < orientation_error:
                orientation_error_wrapped = -(2 * np.pi - orientation_error)
            else:
                orientation_error_wrapped = 2 * np.pi + orientation_error

            state = np.concatenate((state, np.array([orientation_error_wrapped])))

        return state

    def get_reward(self, *args, **kwargs):
        if self.stabilizing:
            return 0

        # possible reward flags: tip_target_dist_squared, tip_target_dist_abs, bonus_at_0.1, bonus_at_0.05
        reward_flags = self.run_config["reward_flags"]

        tip_pos = np.array(self.manipulators[0].get_backbone_positions())[-1][1:]
        target_pos, target_or_quat = p.getBasePositionAndOrientation(self.target_id)
        dist = np.linalg.norm(tip_pos - target_pos[1:])

        reward = 0

        # Todo: throw error when reward_flag exists that isn't supported

        if "tip_target_dist_squared" in reward_flags:
            tip_target_dist_squared_reward = np.square(dist)
            self.reward_component_info[
                "tip_target_dist_squared"
            ] += tip_target_dist_squared_reward
            reward += (
                reward_flags["tip_target_dist_squared"] * tip_target_dist_squared_reward
            )

        if "tip_target_dist" in reward_flags:
            tip_target_dist_reward = dist
            self.reward_component_info["tip_target_dist"] += tip_target_dist_reward
            reward += reward_flags["tip_target_dist"] * tip_target_dist_reward

        if "bonus_at_2" in reward_flags and np.isclose(dist, 0.0, atol=2).all():
            self.reward_component_info["bonus_at_2"] += 1
            reward += reward_flags["bonus_at_2"]

        if "tip_orientation_reward" in reward_flags:
            assert self.target_x_orientation, (
                f"you are requesting to reward tip orientation error, but "
                f"self.target_x_orientation has not been initialized. double check the "
                f"target function you use to assign target position and orientation."
            )

            num_links = len(self.manipulators[0].flexible_joint_indices)
            last_link_state = p.getLinkState(
                self.manipulators[0].bodyUniqueId, num_links - 1
            )
            tip_orientation_x = p.getEulerFromQuaternion(last_link_state[1])[0]
            orientation_error = self.target_x_orientation - tip_orientation_x

            # take care of angle wrapping
            if -np.pi < orientation_error < np.pi:
                orientation_error_wrapped = orientation_error
            elif np.pi < orientation_error:
                orientation_error_wrapped = -(2 * np.pi - orientation_error)
            else:
                orientation_error_wrapped = 2 * np.pi + orientation_error

            orientation_error_wrapped_abs = np.abs(orientation_error_wrapped)
            self.reward_component_info[
                "tip_orientation_reward"
            ] += orientation_error_wrapped_abs
            reward += (
                reward_flags["tip_orientation_reward"] * orientation_error_wrapped_abs
            )

        if "bonus_at_1" in reward_flags and np.isclose(dist, 0.0, atol=1).all():
            self.reward_component_info["bonus_at_1"] += 1
            reward += reward_flags["bonus_at_1"]

        if "bonus_at_0.5" in reward_flags and np.isclose(dist, 0.0, atol=0.5).all():
            self.reward_component_info["bonus_at_0.5"] += 1
            reward += reward_flags["bonus_at_0.5"]

        return reward

    def modify_physics(self):
        if self.get_obstacle_start is not None:
            self.obstacle_pos, self.obstacle_or = self.get_obstacle_start(self)

        if self.get_target_start is not None:
            (
                self.target_start_pos,
                self.target_start_or,
                self.target_x_orientation,
            ) = self.get_target_start(self)

        object_urdf_path = os.path.join(
            os.path.dirname(__file__), "definitions/additional_urdfs/target_sphere.urdf"
        )
        self.target_id = p.loadURDF(
            object_urdf_path,
            self.target_start_pos,
            self.target_start_or,
            useFixedBase=1,
        )

        if self.with_obstacle and self.obstacle_scale:
            obstacle_urdf_path = os.path.join(
                os.path.dirname(__file__), "definitions/additional_urdfs/obstacle.urdf"
            )
            self.obstacle_id = p.loadURDF(
                obstacle_urdf_path,
                self.obstacle_pos,
                self.obstacle_or,
                globalScaling=self.obstacle_scale,
                useFixedBase=1,
            )

    def get_cam_settings(self):
        opt_str = "--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0"
        # cam_width, cam_height = 1920, 1640
        cam_width, cam_height = (
            None,
            None,
        )
        if cam_width is not None and cam_height is not None:
            opt_str += " --width=%d --height=%d" % (cam_width, cam_height)

        cam_distance, cam_yaw, cam_pitch, cam_xyz_target = (
            8.0,
            90.0,
            0.0,
            [0.0, 0.0, 5],
        )

        return opt_str, cam_distance, cam_yaw, cam_pitch, cam_xyz_target

    def check_success(self):
        tip_pos = np.array(self.manipulators[0].get_backbone_positions())[-1][1:]
        target_pos, target_or_quat = p.getBasePositionAndOrientation(self.target_id)
        dist = np.linalg.norm(tip_pos - target_pos[1:])

        return dist <= 1.0
