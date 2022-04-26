import numpy as np
import os

import pybullet as p

from somo.sm_manipulator_definition import SMManipulatorDefinition
from somo.sm_continuum_manipulator import SMContinuumManipulator

from environments.PlanarBlockPushing.env_utils import box_start_funcs

from environments import SomoEnv


class PlanarBlockPushing(SomoEnv.SomoEnv):
    def __init__(self, run_config=None, run_ID=None, render=False, debug=0):
        # todo: visually debug PlanarBlockPushing - something seems off - maybe ground plane too high / in middle of actuator? bad settings for actuator stiffness

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

        start_positions = [[0, 0, 0.5]]
        start_orientations = [p.getQuaternionFromEuler([0, np.pi / 2, np.pi / 2])]
        manipulator_base_constraints = ["static"]

        self.boundary = 16

        action_shape = (4,)

        super().__init__(
            manipulators=manipulators,
            start_positions=start_positions,
            start_orientations=start_orientations,
            manipulator_base_constraints=manipulator_base_constraints,
            active_actuators_list=active_actuators_list,
            action_shape=action_shape,
            run_config=run_config,
            gravity=[0, 0, -196.2],  # scale factor 20 applied to gravity
            ground_plane_height=0,
            run_ID=run_ID,
            render=render,
            debug=debug,
        )

        # set some geometry params
        self.box_start_pos = np.array(self.run_config["box_start_pos"])
        self.box_start_or = np.array(p.getQuaternionFromEuler([0, 0, 0]))
        self.target_pos = np.array(self.run_config["target_pos"])

        try:
            self.get_box_start = box_start_funcs.box_start_funcs[
                self.run_config["box_start_func"]
            ]
        except:
            self.get_box_start = None

    def step(self, action):
        obs, reward, done, info = super().step(action=action)

        try:
            failure_penalty = self.run_config["failure_penalty"]
        except:
            failure_penalty = 0

        if done:
            reward -= failure_penalty
        return (obs, reward, done, info)

    def is_done(self):
        box_pos, _ = p.getBasePositionAndOrientation(
            self.box_id, physicsClientId=self.physics_client
        )
        done = np.linalg.norm(box_pos) >= self.boundary
        return done

    def get_observation_shape(self):
        # possible observation flags: box_pos, box_or, torques
        obs_dimension_per_sample = {
            "box_pos": 2,
            "box_or": 1,
            "box_target_dist_vector": 2,
            "target_pos": 2,
            "box_velocity": 2,
            "positions": 2,
            "velocities": 2,
            "tip_pos": 2,
            "tip_box_dist_vector": 2,
            "manipulator_box_dist": 2,
            "angles": 1,
            "curvatures": 1,
            "applied_input_torques": 4,
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
        box_pos, box_or_quat = p.getBasePositionAndOrientation(
            self.box_id, physicsClientId=self.physics_client
        )
        self.manipulator_box_dist = np.zeros((self.n_manipulators, 3))
        for i, state in enumerate(self.manipulator_states):
            dist_vectors = state["positions"] - np.tile(
                np.array(box_pos), (len(state["positions"]), 1)
            )
            distances = np.linalg.norm(dist_vectors, axis=1)
            closest_link = np.argmin(distances)
            self.manipulator_box_dist[i] = dist_vectors[closest_link]
        state = np.array([])

        if "box_pos" in obs_flags or "box_or" in obs_flags or "box_target_dist_vector":
            box_pos = np.array(box_pos)[:2]
            box_or = np.array([p.getEulerFromQuaternion(box_or_quat)[2]])
            if "box_pos" in obs_flags:
                state = np.concatenate((state, box_pos))
            if "box_or" in obs_flags:
                state = np.concatenate((state, box_or))
            if "box_target_dist_vector" in obs_flags:
                state = np.concatenate((state, box_pos - self.target_pos[:2]))

        if "target_pos" in obs_flags:
            state = np.concatenate((state, self.target_pos[:2]))

        if "box_velocity" in obs_flags:
            box_velocity = np.array(
                p.getBaseVelocity(
                    bodyUniqueId=self.box_id, physicsClientId=self.physics_client
                )[0]
            )
            state = np.concatenate((state, np.array(box_velocity)[:2]))

        if "positions" in obs_flags:
            positions = np.array(
                [state["positions"][:, :2] for state in self.manipulator_states]
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
                [state["velocities"][:, :2] for state in self.manipulator_states]
            )
            if obs_flags["velocities"]:
                velocities = np.array(
                    [
                        self.reduce_state_len(vs, obs_flags["velocities"])
                        for vs in velocities
                    ]
                )
            state = np.concatenate((state, velocities.flatten()))

        if "tip_pos" in obs_flags or "tip_box_dist_vector" in obs_flags:
            positions = np.array(
                [state["positions"][:, :2] for state in self.manipulator_states]
            )
            tip_pos = positions[0][-1]
            if "tip_box_dist_vector" in obs_flags:
                box_pos, _ = p.getBasePositionAndOrientation(
                    self.box_id, physicsClientId=self.physics_client
                )
                state = np.concatenate((state, box_pos[:2] - tip_pos))
            if "tip_pos" in obs_flags:
                state = np.concatenate((state, tip_pos))

        if "manipulator_box_dist" in obs_flags:
            state = np.concatenate((state, self.manipulator_box_dist[:, :2].flatten()))

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

        return state

    def get_reward(self, *args, **kwargs):
        if self.stabilizing:
            return 0
        # TODO: add reward for distance between box and closest backbone link
        # possible reward flags: tip_box_dist_squared, tip_box_dist_abs, box_target_dist_squared, box_target_dist, bonus_at_0.1, bonus_at_0.05
        reward_flags = self.run_config["reward_flags"]

        tip_pos = np.array(self.manipulators[0].get_backbone_positions())[-1][:2]
        box_pos, box_or_quat = p.getBasePositionAndOrientation(
            self.box_id, physicsClientId=self.physics_client
        )
        tip_box_dist = np.linalg.norm(tip_pos - box_pos[:2])
        box_target_dist = np.linalg.norm(box_pos[:2] - self.target_pos[:2])

        reward = 0

        if "manipulator_box_dist" in reward_flags:
            manipulator_box_dist_reward = np.sum(
                np.linalg.norm(self.manipulator_box_dist, axis=1)
            )
            # print(np.linalg.norm(self.manipulator_box_dist, axis=1))
            self.reward_component_info[
                "manipulator_box_dist"
            ] += manipulator_box_dist_reward
            reward += reward_flags["manipulator_box_dist"] * manipulator_box_dist_reward

        if "tip_box_dist" in reward_flags:
            tip_box_dist_reward = tip_box_dist
            self.reward_component_info["tip_box_dist"] += tip_box_dist_reward
            reward += reward_flags["tip_box_dist"] * tip_box_dist_reward

        if "box_target_dist_squared" in reward_flags:
            box_target_dist_squared_reward = np.square(box_target_dist).sum()
            self.reward_component_info[
                "box_target_dist_squared"
            ] += box_target_dist_squared_reward
            reward += (
                reward_flags["box_target_dist_squared"] * box_target_dist_squared_reward
            )

        if "box_target_dist" in reward_flags:
            box_target_dist_reward = box_target_dist
            self.reward_component_info["box_target_dist"] += box_target_dist_reward
            reward += reward_flags["box_target_dist"] * box_target_dist_reward

        if (
            "bonus_at_2" in reward_flags
            and np.isclose(box_target_dist, 0.0, atol=2).all()
        ):
            self.reward_component_info["bonus_at_2"] += 1
            reward += reward_flags["bonus_at_2"]

        if (
            "bonus_at_1" in reward_flags
            and np.isclose(box_target_dist, 0.0, atol=1).all()
        ):
            self.reward_component_info["bonus_at_1"] += 1
            reward += reward_flags["bonus_at_1"]

        if (
            "bonus_at_0.5" in reward_flags
            and np.isclose(box_target_dist, 0.0, atol=0.5).all()
        ):
            self.reward_component_info["bonus_at_0.5"] += 1
            reward += reward_flags["bonus_at_0.5"]

        return reward

    def modify_physics(self):
        if self.get_box_start is not None:
            self.box_start_pos, self.box_start_or = self.get_box_start(self)

        object_urdf_path = os.path.join(
            os.path.dirname(__file__), "definitions/additional_urdfs/cube.urdf"
        )

        self.box_id = p.loadURDF(
            object_urdf_path,
            self.box_start_pos,
            self.box_start_or,
            useFixedBase=0,
            physicsClientId=self.physics_client,
        )
        p.changeDynamics(
            self.box_id, -1, lateralFriction=2, physicsClientId=self.physics_client
        )

        object_urdf_path = os.path.join(
            os.path.dirname(__file__), "definitions/additional_urdfs/cube_target.urdf"
        )

        self.target_id = p.loadURDF(
            object_urdf_path,
            self.target_pos,
            p.getQuaternionFromEuler(
                [0, 0, 0]
            ),  # todo: consider making it possible to change box target orientation
            useFixedBase=1,
            physicsClientId=self.physics_client,
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
            10.0,
            0.0,
            -88.0,
            [0.0, 6.0, 0.1],
        )

        return opt_str, cam_distance, cam_yaw, cam_pitch, cam_xyz_target

    def check_success(self):
        box_pos, box_or_quat = p.getBasePositionAndOrientation(
            self.box_id, physicsClientId=self.physics_client
        )
        box_target_dist = np.linalg.norm(box_pos[:2] - self.target_pos[:2])

        return box_target_dist <= 1.0
