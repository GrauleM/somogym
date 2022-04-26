# TODO MORITZ: fix somo urdf generation - we should not be spawning the same finger in root every time :P
import os
import numpy as np
import copy

import pybullet as p

from somo.sm_manipulator_definition import SMManipulatorDefinition
from somo.sm_continuum_manipulator import SMContinuumManipulator

from environments import SomoEnv

# TODO URGENT: EXTERNAL FORCE DOES NOT WORK HERE: USE NEW APPROACH IN ALL ENVS FROM filament-envs BRANCH
class AntipodalGripper(SomoEnv.SomoEnv):
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

        # MAJOR TODO: round all box corners
        # adjustable settings / parameters
        self.max_base_z_displacement_per_step = 0.005
        self.max_base_height = 5
        self.spring_stiffness = 1  # todo: tune this to make sure spring stiffness and gravity force are in same order of magnitude
        self.n_manipulators = 2

        self.planar_only = run_config["planar_only"]
        if self.planar_only:  # todo: write separate tests for planar only AND full env
            action_shape = (5,)
            self.z_pos_index = (
                4  # the index within action of the action that changes the palm height
            )
            active_actuators_list = [
                (
                    [
                        0,
                        1,
                    ],
                    [
                        0,
                        0,
                    ],
                )
            ] * self.n_manipulators
            filename = "SG_finger_planar.yaml"
        else:
            action_shape = (9,)
            self.z_pos_index = (
                8  # the index within action of the action that changes the palm height
            )
            active_actuators_list = [([0, 0, 1, 1], [0, 1, 0, 1])] * self.n_manipulators
            filename = "SG_finger.yaml"

        # load the manipulator definition
        manipulator_def_file = os.path.join(
            os.path.dirname(__file__), "definitions/", filename
        )
        manipulator_def = SMManipulatorDefinition.from_file(manipulator_def_file)
        manipulator_def.urdf_filename = os.path.join(
            os.path.dirname(__file__),
            "definitions/" + manipulator_def.manipulator_name + ".urdf",
        )

        manipulators = [
            SMContinuumManipulator(manipulator_def) for i in range(self.n_manipulators)
        ]

        manipulator_base_constraints = ["constrained"] * self.n_manipulators

        self.base_height = 2.75
        self.base_width = 1
        self.finger_angle = 10

        start_positions = [
            [0, -self.base_width / 2, self.base_height],
            [0, self.base_width / 2, self.base_height],
        ]
        start_orientations = [
            p.getQuaternionFromEuler(np.deg2rad([self.finger_angle, 180, 0])),
            p.getQuaternionFromEuler(np.deg2rad([self.finger_angle, 180, 180])),
        ]

        self.base_positions_start = copy.deepcopy(start_positions)
        self.base_orientations_start = copy.deepcopy(start_orientations)

        self.base_positions = copy.deepcopy(self.base_positions_start)
        self.base_orientations = copy.deepcopy(self.base_orientations_start)

        # set some geometry params
        self.box_start_pos = np.array([0.0, 0.0, 0.5])
        self.box_start_or = np.array([0, 0, 0])
        self.box_start_or_quat = p.getQuaternionFromEuler(list(self.box_start_or))
        self.prev_box_pos = self.box_start_pos
        self.prev_box_or = self.box_start_or

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

    def step(self, action):

        # todo: xx careful: if this action is applied this way, it cannot be ramped according to torque limits

        # MOVE THE ACTUATOR BASE
        position_change = [
            0,
            0,
            action[self.z_pos_index] * self.max_base_z_displacement_per_step,
        ]  # change position by a scaled amount
        orientation_change = [0, 0, 0]

        for i in range(self.n_manipulators):
            self.base_positions[i] = [
                x + y for x, y in zip(self.base_positions[i], position_change)
            ]
            # clip base z height
            self.base_positions[i][-1] = (
                self.max_base_height
                if self.base_positions[i][-1] > self.max_base_height
                else self.base_positions[i][-1]
            )

        # moves the base by changing it's position constraint - may be not great practice, but works as long as displacements are small enough
        for manipulator, position, orientation in zip(
            self.manipulators, self.base_positions, self.base_orientations
        ):
            p.changeConstraint(
                manipulator.baseConstraintUniqueId,
                position,
                orientation,
                physicsClientId=self.physics_client,
            )

        # apply an additional downwards force to the box that is a function of box height; this way, the higher we lift the box, the better the grasping force

        box_pos, _ = p.getBasePositionAndOrientation(
            self.box_id, physicsClientId=self.physics_client
        )
        box_height = box_pos[2]
        # print(f"BOX HEIGHT: {box_height}")
        force = [
            0,
            0,
            -self.spring_stiffness * box_height,
        ]  # URENT TODO: make sure sign is right here and everywhere else

        p.applyExternalForce(
            objectUniqueId=self.box_id,
            linkIndex=-1,
            forceObj=force,
            posObj=box_pos,
            flags=p.WORLD_FRAME,
            physicsClientId=self.physics_client,
        )

        self.applied_height = position_change[-1]

        # APPLY ACTUATOR TORQUES to the active actuators and finally TAKE the physics STEPs
        obs, reward, done, info = super().step(action=action)

        # todo: add final evaluation / grasp movement here; force palm up for n steps and record reward. make this an optional thing for this env (can either move up at the end or not)
        return (obs, reward, done, info)

    def get_observation_shape(self):
        # possible observation flags: box_pos, box_or, torques
        num_torques = self.action_space.shape[0] - 1
        workspace_dim = 3 - int(self.planar_only)
        obs_dimension_per_sample = {
            "box_pos": 3,
            "box_or": 3,
            "box_velocity": 3,
            "positions": workspace_dim * 2,
            "velocities": workspace_dim * 2,
            "tip_pos": workspace_dim * 2,
            "manipulator_box_dist": workspace_dim * 2,
            "angles": 1 * 2,
            "curvatures": 1 * 2,
            "applied_input_torques": num_torques,
            "applied_input_height": 1,
        }
        obs_flags = self.run_config["observation_flags"]
        obs_len = 0
        for f in obs_flags:
            num_pts = obs_flags[f] if obs_flags[f] is not None else 1
            obs_len += num_pts * obs_dimension_per_sample[f]
        return (obs_len,)

    def get_observation(self):
        # todo: add contact points in observation
        obs_flags = self.run_config["observation_flags"]
        workspace_dim = 3 - int(self.planar_only)
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

        if "box_pos" in obs_flags or "box_or" in obs_flags:
            if "box_pos" in obs_flags:
                state = np.concatenate((state, np.array(box_pos)))
            if "box_or" in obs_flags:
                box_or = p.getEulerFromQuaternion(box_or_quat)
                state = np.concatenate((state, np.array(box_or)))

        if "box_velocity" in obs_flags:
            box_velocity = np.array(
                p.getBaseVelocity(
                    bodyUniqueId=self.box_id, physicsClientId=self.physics_client
                )[0]
            )
            state = np.concatenate((state, np.array(box_velocity)))

        if "positions" in obs_flags:
            positions = np.array(
                [
                    state["positions"][:, int(self.planar_only) :]
                    for state in self.manipulator_states
                ]
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
                [
                    state["velocities"][:, int(self.planar_only) :]
                    for state in self.manipulator_states
                ]
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
                [
                    state["positions"][:, int(self.planar_only) :]
                    for state in self.manipulator_states
                ]
            )
            tip_pos = positions[:, -1]
            state = np.concatenate((state, tip_pos.flatten()))

        if "manipulator_box_dist" in obs_flags:
            state = np.concatenate(
                (state, self.manipulator_box_dist[:, int(self.planar_only) :].flatten())
            )

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
            applied_input_torques = np.array(self.applied_torque)[:-1]
            state = np.concatenate((state, applied_input_torques.flatten()))

        if "applied_input_height" in obs_flags:
            state = np.concatenate((state, np.array([self.applied_height])))

        return state

    def get_reward(self):
        if (
            self.stabilizing
        ):  # todo: move this to SomoEnv base class; also introduce attribute to set flexible stabilization time
            return 0

        reward_flags = self.run_config["reward_flags"]
        box_pos, _ = p.getBasePositionAndOrientation(
            self.box_id, physicsClientId=self.physics_client
        )
        box_height = box_pos[2]

        reward = 0

        if "box_height" in reward_flags:
            box_height_reward = box_height
            self.reward_component_info["box_height"] += box_height
            reward += reward_flags["box_height"] * box_height

        if "manipulator_box_dist" in reward_flags:
            manipulator_box_dist_reward = np.sum(
                np.linalg.norm(self.manipulator_box_dist, axis=1)
            )
            self.reward_component_info[
                "manipulator_box_dist"
            ] += manipulator_box_dist_reward
            reward += reward_flags["manipulator_box_dist"] * manipulator_box_dist_reward

        return reward

    def modify_physics(self):

        self.base_positions = copy.deepcopy(self.base_positions_start)
        self.base_orientations = copy.deepcopy(self.base_orientations_start)

        # todo: this should be part of somo definitions and applied automatically; update somo
        for manipulator in self.manipulators:
            contact_properties = {
                "lateralFriction": 3.0,
            }
            manipulator.set_contact_property(contact_properties)

        # define box to manipulate # todo: add __file__ and relative path to make this path safer for all envs
        box_urdf_path = os.path.join(
            os.path.dirname(__file__), "definitions/additional_urdfs/cube.urdf"
        )

        # todo: define meaningful weight and inertias for the box / other boxs

        self.box_id = p.loadURDF(
            box_urdf_path,
            self.box_start_pos,
            self.box_start_or_quat,
            useFixedBase=0,
            physicsClientId=self.physics_client,
        )
        p.changeDynamics(
            self.box_id, -1, lateralFriction=0.5, physicsClientId=self.physics_client
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
            6.0,
            -90.0,
            -22.0,
            [0.0, 0.0, 1.5],
        )

        return opt_str, cam_distance, cam_yaw, cam_pitch, cam_xyz_target

    def check_success(self):
        box_pos, box_or_quat = p.getBasePositionAndOrientation(
            self.box_id, physicsClientId=self.physics_client
        )

        return box_pos[2] >= 2
