# TODO MORITZ: fix somo urdf generation - we should not be spawning the same finger in root every time :P
import os
import numpy as np
from pyquaternion import Quaternion

import pybullet as p

from somo.sm_manipulator_definition import SMManipulatorDefinition
from somo.sm_continuum_manipulator import SMContinuumManipulator

from environments import SomoEnv


class PenSpinner(SomoEnv.SomoEnv):
    """
    TODO: replace this with actual description in this format
    Description:
        An antipodal gripper with 2 continuum/soft fingers
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

    Actions:
        Type: Continuous
        Num   Action
        0     Push cart to the left
        1     Push cart to the right
        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.

    """

    def __init__(
        self,
        run_config=None,
        run_ID=None,
        render=False,
        debug=0,  # todo: this will be removed / merged into render flag. properly work with render mode. change debug panel to show each component of the reward in real-time. reserve debug panel for behavior cloning; remove from standard SomoEnv
    ):

        # adjustable settings / parameters
        self.n_manipulators = 5

        # load the manipulator definition
        manipulator_def_file = os.path.join(
            os.path.dirname(__file__), "definitions/PS_finger_round.yaml"
        )
        manipulator_def = SMManipulatorDefinition.from_file(manipulator_def_file)
        manipulator_def.urdf_filename = os.path.join(
            os.path.dirname(__file__),
            "definitions/" + manipulator_def.manipulator_name + ".urdf",
        )

        active_actuators_list = [([0, 0, 1, 1], [0, 1, 0, 1])] * self.n_manipulators

        manipulators = [
            SMContinuumManipulator(manipulator_def) for _ in range(self.n_manipulators)
        ]

        manipulator_base_constraints = ["static"] * self.n_manipulators

        base_height = 2.5
        grasp_width = 1.2
        finger_spacing = 0.8
        pen_radius = 0.4

        start_positions = [
            [0, -grasp_width / 2, base_height],
            [0, grasp_width / 2, base_height],
            [-finger_spacing, grasp_width / 2, base_height],
            [finger_spacing, grasp_width / 2, base_height],
            [2 * finger_spacing, grasp_width / 2, base_height],
        ]
        start_orientations = [
            p.getQuaternionFromEuler(np.deg2rad([0, 180, 0])),
            p.getQuaternionFromEuler(np.deg2rad([0, 180, 180])),
            p.getQuaternionFromEuler(np.deg2rad([0, 180, 180])),
            p.getQuaternionFromEuler(np.deg2rad([0, 180, 180])),
            p.getQuaternionFromEuler(np.deg2rad([0, 180, 180])),
        ]

        self.base_positions = start_positions
        self.base_orientations = start_orientations

        action_shape = (20,)

        # set some geometry params

        self.obj_start_pos = np.array([0.0, 0.0, pen_radius + 0.01])
        self.obj_start_or = np.array([0, np.pi / 2, 0])
        self.obj_start_or_quat = p.getQuaternionFromEuler(list(self.obj_start_or))
        self.prev_obj_pos = self.obj_start_pos
        self.prev_obj_or = self.obj_start_or

        # set the possible range of object target poses; the env randomly samples from these at the beginning of each episode and
        # then tries to move the object to this pose
        if "target_position_delta_range" in run_config.keys():
            target_position_delta_range = run_config["target_position_delta_range"]
        else:
            target_position_delta_range = {
                "min": [0.0, 0.0, 0.0],
                "max": [0.0, 0.0, 0.0],
            }
        if "target_orientation_delta_range" in run_config.keys():
            target_orientation_delta_range = run_config[
                "target_orientation_delta_range"
            ]
        else:
            target_orientation_delta_range = {
                "min": [0.0, -0.1, 0.2],
                "max": [0.0, 0.1, 0.3],
            }

        self.target_position_delta_range = target_position_delta_range
        self.target_orientation_delta_range = target_orientation_delta_range

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

    def get_observation_shape(self):
        obs_dimension_per_sample = {
            "object_pos": 3,
            "object_or": 3,  # in Euler angles
            "object_or_quat": 4,  # in quaternions
            "target_position": 3,
            "target_orientation": 3,  # in Euler angles
            "target_orientation_quat": 4,  # ... in quaternions
            "position_error": 3,
            "orientation_error": 3,  # ... in Euler angles
            "orientation_error_quat": 4,  # ... in quaternions
            "object_velocity": 3,
            "positions": 3 * 5,
            "velocities": 3 * 5,
            "tip_pos": 3 * 5,
            "angles": 1 * 5,
            "curvatures": 1 * 5,
            "applied_input_torques": 4 * 5,
            "tip_pos_error_vector": 6,
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

        object_pos, object_or_quat = p.getBasePositionAndOrientation(
            self.object_id, physicsClientId=self.physics_client
        )
        object_or = p.getEulerFromQuaternion(object_or_quat)

        targ_pos, targ_or_quat = self.target_position, self.target_orientation
        targ_or = p.getEulerFromQuaternion(targ_or_quat)

        if "object_pos" in obs_flags:
            state = np.concatenate((state, np.array(object_pos)))
        if "object_or" in obs_flags:
            state = np.concatenate((state, np.array(object_or)))
        if "object_or_quat" in obs_flags:
            state = np.concatenate((state, np.array(object_or_quat)))

        if "target_position" in obs_flags:
            state = np.concatenate((state, np.array(targ_pos)))
        if "target_orientation" in obs_flags:
            state = np.concatenate((state, np.array(targ_or)))
        if "target_orientation_quat" in obs_flags:
            state = np.concatenate((state, np.array(targ_or_quat)))

        if "position_error" in obs_flags:
            state = np.concatenate((state, np.array(object_pos) - np.array(targ_pos)))
        if "orientation_error" in obs_flags:
            state = np.concatenate(
                (state, np.array(object_or) - np.array(targ_or))
            )  # note: not a great way to compute orientation distances bcs angle wrapping and so on
        if "orientation_error_quat" in obs_flags:
            q1 = Quaternion(
                object_or_quat[3],
                object_or_quat[0],
                object_or_quat[1],
                object_or_quat[2],
            )
            q2 = Quaternion(
                self.target_orientation[3],
                self.target_orientation[0],
                self.target_orientation[1],
                self.target_orientation[2],
            )
            orientation_error_quat = q1.inverse * q2
            orientation_error = np.array(
                [
                    orientation_error_quat[3],
                    orientation_error_quat[0],
                    orientation_error_quat[1],
                    orientation_error_quat[2],
                ]
            )
            state = np.concatenate((state, orientation_error))

        if "object_velocity" in obs_flags:
            object_velocity = np.array(
                p.getBaseVelocity(
                    bodyUniqueId=self.object_id, physicsClientId=self.physics_client
                )[0]
            )
            state = np.concatenate((state, np.array(object_velocity)))

        if "positions" in obs_flags:
            positions = np.array(
                [state["positions"] for state in self.manipulator_states]
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
                [state["velocities"] for state in self.manipulator_states]
            )
            if obs_flags["velocities"]:
                velocities = np.array(
                    [
                        self.reduce_state_len(vs, obs_flags["velocities"])
                        for vs in velocities
                    ]
                )
            state = np.concatenate((state, velocities.flatten()))

        if "tip_pos" in obs_flags:
            positions = np.array(
                [state["positions"] for state in self.manipulator_states]
            )
            tip_pos = positions[:, -1]
            state = np.concatenate((state, tip_pos.flatten()))

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

        if "tip_pos_error_vector" in obs_flags:

            end_pos_target = np.array(
                p.getLinkState(
                    bodyUniqueId=self.target_id,  # todo: why unexpected argument error?
                    linkIndex=0,
                    physicsClientId=self.physics_client,
                )[0]
            )
            end_pos_object = np.array(
                p.getLinkState(
                    bodyUniqueId=self.object_id,
                    linkIndex=0,
                    physicsClientId=self.physics_client,
                )[0]
            )

            end_pos_error_vec = end_pos_target - end_pos_object

            tip_pos_target = np.array(
                p.getLinkState(
                    bodyUniqueId=self.target_id,
                    linkIndex=1,
                    physicsClientId=self.physics_client,
                )[0]
            )
            tip_pos_object = np.array(
                p.getLinkState(
                    bodyUniqueId=self.object_id,
                    linkIndex=1,
                    physicsClientId=self.physics_client,
                )[0]
            )

            tip_pos_error_vec = tip_pos_target - tip_pos_object

            state = np.concatenate(
                (state, end_pos_error_vec.flatten(), tip_pos_error_vec.flatten())
            )

        return state

    def get_reward(self, action=None):

        # TODO: orientation rewards is NOT working correctly with quaternions; find out why and how it can be fixed
        # TODO: while orientation rewards is not working, implement/use simpler reward that is a line of points along the pen and uses position error

        if (
            self.stabilizing
        ):  # todo: move this to SomoEnv base class; also introduce attribute to set flexible stabilization time
            return 0

        # possible reward flags for this env: orientation, position
        reward_flags = self.run_config["reward_flags"]

        reward = 0
        # note: there are some challenges associated with using the quaternion distance for orientation error, see for example https://math.stackexchange.com/questions/90081/quaternion-distance
        if "orientation_quat" in reward_flags or "position" in reward_flags:
            obj_pos, obj_or_quat = p.getBasePositionAndOrientation(
                self.object_id, physicsClientId=self.physics_client
            )

            if "position" in reward_flags:
                position_reward = np.linalg.norm(
                    np.array(obj_pos) - self.target_position
                )
                self.reward_component_info["position"] += position_reward
                reward += reward_flags["position"] * position_reward

            if "orientation_quat" in reward_flags:
                # using the pyquaternion module to get distance between target and object orientation
                q1 = Quaternion(
                    obj_or_quat[3], obj_or_quat[0], obj_or_quat[1], obj_or_quat[2]
                )
                q2 = Quaternion(
                    self.target_orientation[3],
                    self.target_orientation[0],
                    self.target_orientation[1],
                    self.target_orientation[2],
                )
                orientation_reward = Quaternion.absolute_distance(
                    q1.normalised, q2.normalised
                )

                # orientation_reward
                # import pdb
                # pdb.set_trace()
                # consider also working with other quaternion distances
                self.reward_component_info["orientation_quat"] += orientation_reward
                reward += reward_flags["orientation_quat"] * orientation_reward

        if "tip_end_position_error" in reward_flags:

            end_pos_target = np.array(
                p.getLinkState(
                    bodyUniqueId=self.target_id,
                    linkIndex=0,
                    physicsClientId=self.physics_client,
                )[0]
            )
            end_pos_object = np.array(
                p.getLinkState(
                    bodyUniqueId=self.object_id,
                    linkIndex=0,
                    physicsClientId=self.physics_client,
                )[0]
            )

            tip_pos_target = np.array(
                p.getLinkState(
                    bodyUniqueId=self.target_id,
                    linkIndex=1,
                    physicsClientId=self.physics_client,
                )[0]
            )
            tip_pos_object = np.array(
                p.getLinkState(
                    bodyUniqueId=self.object_id,
                    linkIndex=1,
                    physicsClientId=self.physics_client,
                )[0]
            )

            tip_end_position_error = np.linalg.norm(
                end_pos_target - end_pos_object
            ) + np.linalg.norm(tip_pos_target - tip_pos_object)

            self.reward_component_info[
                "tip_end_position_error"
            ] += tip_end_position_error
            reward += reward_flags["tip_end_position_error"] * tip_end_position_error

        return reward

    def modify_physics(self):

        # todo: this should be part of somo definitions and applied automatically; update somo
        for manipulator in self.manipulators:
            contact_properties = {
                "lateralFriction": 3,
                "restitution": 0.7,
            }
            manipulator.set_contact_property(contact_properties)

        # define object to manipulate # todo: add __file__ and relative path to make this path safer for all envs
        object_urdf_path = os.path.join(
            os.path.dirname(__file__), "definitions/additional_urdfs/pen.urdf"
        )

        # todo: define meaningful weight and inertias for the pen / other objects

        self.object_id = p.loadURDF(
            object_urdf_path,
            self.obj_start_pos,
            self.obj_start_or_quat,
            useFixedBase=0,
            physicsClientId=self.physics_client,
        )

        p.changeDynamics(
            self.object_id, -1, lateralFriction=1.0, physicsClientId=self.physics_client
        )

        # define target pose for the object
        target_urdf_path = os.path.join(
            os.path.dirname(__file__), "definitions/additional_urdfs/pen_target.urdf"
        )

        assert (
            self.np_random
        ), f"for this environment, seeding is required before running any steps. Seed with env.seed(seed_nr)."

        self.target_position = (
            self.obj_start_pos
            + self.np_random.uniform(
                self.target_position_delta_range["min"],
                self.target_position_delta_range["max"],
            )
        ).tolist()
        self.target_orientation = p.getQuaternionFromEuler(
            (
                self.obj_start_or
                + self.np_random.uniform(
                    self.target_orientation_delta_range["min"],
                    self.target_orientation_delta_range["max"],
                )
            ).tolist()
        )

        self.target_id = p.loadURDF(
            target_urdf_path,
            self.target_position,
            self.target_orientation,
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
            4.0,
            45.0,
            -42.0,
            [0.0, 0.50, 0.0],
        )

        return opt_str, cam_distance, cam_yaw, cam_pitch, cam_xyz_target

    def check_success(self):
        end_pos_target = np.array(
            p.getLinkState(
                bodyUniqueId=self.target_id,
                linkIndex=0,
                physicsClientId=self.physics_client,
            )[0]
        )
        end_pos_object = np.array(
            p.getLinkState(
                bodyUniqueId=self.object_id,
                linkIndex=0,
                physicsClientId=self.physics_client,
            )[0]
        )

        tip_pos_target = np.array(
            p.getLinkState(
                bodyUniqueId=self.target_id,
                linkIndex=1,
                physicsClientId=self.physics_client,
            )[0]
        )
        tip_pos_object = np.array(
            p.getLinkState(
                bodyUniqueId=self.object_id,
                linkIndex=1,
                physicsClientId=self.physics_client,
            )[0]
        )

        tip_error = np.array(
            [
                np.linalg.norm(end_pos_target - end_pos_object),
                np.linalg.norm(tip_pos_target - tip_pos_object),
            ]
        )

        return np.max(tip_error) <= 0.25
