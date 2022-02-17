import os
import numpy as np
from copy import deepcopy

import gym
from gym.utils import seeding
from gym import spaces

import pybullet as p

from environments.utils.debug_dashboard import Debugger

# TODO: major: add ability to have dynamics noise, action noise, disturbances, observation noise
class SomoEnv(gym.Env):
    def __init__(
        self,
        manipulators,
        start_positions,
        start_orientations,
        manipulator_base_constraints,
        active_actuators_list,  # format: [([act nrs of active actuators for manip0],[axis inds of active actuatorsfor manip0]),([...manip1])] # todo: this is a bit cumbersome, refactor?
        action_shape,
        run_config,
        gravity,
        ground_plane_height=None,
        run_ID=None,
        render=False,
        debug=False,
        manipulator_load_flags=None,
    ):

        # todo: document

        # todo: rearrange attribute assignments to match input into init
        self.run_config = run_config
        self.run_ID = run_ID

        self.render_gui = render
        self.debug = debug
        self.debugger = Debugger(self, sections=debug) if debug else None
        self.gravity = gravity
        self.ground_plane_height = ground_plane_height

        # initialize all general params
        self.np_random = None
        self.ep_count = 0
        self.step_count = 0
        self.first_run_done = False
        self.applied_torque = None
        self.stabilizing = True

        # while we're unsure, let's group self. ... assignments in meaningful way but keep them as before
        self.action_time = self.run_config[
            "action_time"
        ]  # time over which an applied action is held constant in seconds
        self.bullet_time_step = self.run_config[
            "bullet_time_step"
        ]  # time for the pybullet simulator steps in seconds

        # max_torque_rate is the maximum rate at which changes in torque can be applied. max_torque_rate should be either a scalar or a list of length len(action space).
        if isinstance(self.run_config["max_torque_rate"], list):
            self.max_torque_rate = np.array(self.run_config["max_torque_rate"])
        else:
            self.max_torque_rate = np.array(
                [self.run_config["max_torque_rate"]] * action_shape[0]
            )

        # a torque multiplier is introduced to ensure normalized actions. torque multiplier should be either a scalar or a list of length len(action space). # todo: document what this means
        if isinstance(self.run_config["torque_multiplier"], list):
            self.torque_multiplier = np.array(self.run_config["torque_multiplier"])
        else:
            self.torque_multiplier = np.array(
                [self.run_config["torque_multiplier"]] * action_shape[0]
            )

        self.manipulators = manipulators
        self.n_manipulators = len(manipulators)
        self.start_positions = start_positions
        self.start_orientations = start_orientations
        self.active_actuators_list = active_actuators_list  #         # example for active_actuators entry for each manipulator:   ([0,0,1,1,2,2,3,3],  [0,1,0,1,0,1,0,1]); active_actuators_list is a list of these entries
        self.manipulator_base_constraints = manipulator_base_constraints

        if manipulator_load_flags:
            self.manipulator_load_flags = manipulator_load_flags
        else:
            self.manipulator_load_flags = [None] * len(
                manipulator_base_constraints
            )  # there a None into a list of None's for the manipulator flags

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=action_shape,
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.get_observation_shape(),
            dtype=np.float32,
        )

        self.validate()
        self.reset_reward_component_info = {}
        if self.run_config["reward_flags"] is not None:
            self.reset_reward_component_info = dict(
                (flag, 0) for flag in self.run_config["reward_flags"].keys()
            )
        self.steps_beyond_done = None
        # todo: explain each entry in run config in the readme / doc

    def validate(self):
        assert (
            len(self.manipulators) == len(self.start_positions)
            and len(self.manipulators) == len(self.start_orientations)
            and len(self.manipulators) == len(self.manipulator_base_constraints)
        ), "The number of manipulators has to match the number of start positions, start orientations, and manipulator constraints provided"

        # make sure that the number of actions is >= the number of actuated actuators
        assert (
            len(self.active_actuators_list) <= self.action_space.shape[0]
        ), f"You provided {len(self.active_actuators_list[0])} active actuators, but the action space has shape {self.action_space.shape}."

        # make sure the actuator indices and actuator axis indices are of the same length
        for i in range(len(self.manipulators)):
            assert len(self.active_actuators_list[i][0]) == len(
                self.active_actuators_list[i][1]
            ), f"You provided {len(self.active_actuators_list[i][0])} active actuator indices for manipulator {i}, and {len(self.active_actuators_list[i][1])} active axes"

        assert (
            len(self.max_torque_rate) == self.action_space.shape[0]
        ), f"max_torque_rate shape and action space shape should match"
        assert (
            len(self.torque_multiplier) == self.action_space.shape[0]
        ), f"torque multiplier shape and action space shape should match"

    def get_manipulator_states(
        self, components=("positions", "velocities", "angles", "curvatures")
    ):
        # todo: should this optionally take a list of components to return to reduce unecessary pybullet queries?
        states = [None] * self.n_manipulators
        for i, manipulator in enumerate(self.manipulators):
            state = {}
            (
                positions,
                velocities,
            ) = manipulator.get_backbone_link_positions_and_velocities()
            state["positions"] = np.array(positions)
            state["velocities"] = np.array(velocities)
            if "angles" in components:
                state["angles"] = np.array(manipulator.get_backbone_angles())
            if "curvatures" in components:
                state["curvatures"] = np.array(manipulator.get_backbone_curvatures())
            # TODO: total actuator torques (spring + applied) per joint.
            state["step_count"] = self.step_count
            states[i] = deepcopy(state)
        return states

    def reduce_state_len(self, state_array, n_samples):
        len_state = len(state_array)
        # if len_state % n_samples != 0:
        #     print(f"WARNING: {len_state} length state cannot be evenly split into {n_samples} samples.")
        if n_samples == 1:
            # return last state element if n_samples = 1
            return np.array([state_array[-1]])
        assert n_samples > 1, f"n_samples {n_samples} is not >= 1"
        sample_inds = np.flip(
            np.linspace(len_state - 1, 0, num=n_samples, endpoint=False, dtype=int)
        )
        reduced_state = np.zeros(([n_samples] + list(state_array.shape)[1:]))
        for i, ind in enumerate(sample_inds):
            reduced_state[i] = state_array[ind]
        return reduced_state

    def get_observation_shape(self):
        raise NotImplementedError("observation_shape function not implemented")

    def get_observation(self):
        # todo: explore whether normalization of observations has an effect on training
        raise NotImplementedError("observation function not implemented")

    def get_reward(self):
        raise NotImplementedError("reward function not implemented")

    def modify_physics(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        # we have to propagate the seed for action space and observation space to make sure all steps are repeatable / affected by seeding
        self.action_space.seed(seed=seed)
        self.observation_space.seed(seed=seed)

    def step(self, action, external_forces=None):

        assert self.action_space.contains(
            action
        ), f"action {action} not in action space"
        assert (
            self.applied_torque is not None
        ), f"previous action is {self.applied_torque}"

        # ramp to new action / torque
        delta_torque = action * self.torque_multiplier - self.applied_torque
        ramp_sign = np.sign(np.array(delta_torque))
        # torque rate = torque/sec, bullet_time_step = sec/sim_step
        max_torque_per_sim_step = self.max_torque_rate * self.bullet_time_step

        reward = 0
        # todo document: state that the first n elements are for continuum manipulators, and the remaining actions are for remaining actuators

        for i in range(0, int(np.ceil(self.action_time / self.bullet_time_step))):

            applied_delta_torque = np.minimum(
                max_torque_per_sim_step * (i + 1), np.abs(delta_torque)
            )
            applied_torque = self.applied_torque + ramp_sign * applied_delta_torque

            # apply action to manipulator
            action_ind = 0  # counts the number of actions that have been applied to actuators so far

            for manipulator, active_actuators_per_manipulator in zip(
                self.manipulators, self.active_actuators_list
            ):
                action_len = len(
                    active_actuators_per_manipulator[0]
                )  # number of actions to be applied in this loop iteration (i.e., the input dim of the manipulator that is currently being addressed)
                manipulator.apply_actuation_torques(
                    actuator_nrs=active_actuators_per_manipulator[0],
                    axis_nrs=active_actuators_per_manipulator[
                        1
                    ],  # todo: have to change this; solution: add somo method.
                    actuation_torques=list(
                        applied_torque[action_ind : action_ind + action_len]
                    ),
                )
                action_ind += action_len
            # xx urgent todo for somo core: evaluate if and when and why it matters when 0 torques are applied and understand the effect of applying torques in a loop vs at once

            # add external forces if provided:
            if external_forces:
                for external_force in external_forces:
                    p.applyExternalForce(
                        objectUniqueId=external_force[0],
                        linkIndex=external_force[1],
                        forceObj=external_force[2],
                        posObj=external_force[3],
                        flags=external_force[4],
                        physicsClientId=self.physics_client,
                    )

            p.stepSimulation(self.physics_client)

        self.applied_torque = applied_torque
        observation = self.get_observation()
        reward += self.get_reward()
        info = self.reward_component_info

        if self.debug:
            self.debugger.update(
                reward=reward, action=action, observation=observation
            )  # XX already gets step, reward_component_info, applied_torque; needs reward, action, observation
        done = self.is_done()
        if done and self.steps_beyond_done is None:
            self.steps_beyond_done = 0
        elif done and self.steps_beyond_done == 0:
            print(
                "WARNING: "
                "You are calling 'step()' even though this "
                "environment has already returned done = True. You "
                "should always call 'reset()' once you receive 'done = "
                "True' -- any further steps are undefined behavior."
            )
            self.steps_beyond_done += 1

        self.step_count += 1
        return (observation, reward, done, info)

    def is_done(self):
        return False

    def reset(self, run_render=False):

        run_render = run_render or self.render_gui

        # Log info to console
        print(f"\n\n\nENV RESET: episode {self.ep_count}.")
        if self.run_ID:
            print(f"- Run ID: {self.run_ID}.\n\n\n")
        print("\n\n\n")

        # TODO: rename to "physics_instantiated" or "simulator_instantiated" or similar
        if not self.first_run_done:
            if run_render:
                # make rendering prettier
                (
                    opt_str,
                    cam_distance,
                    cam_yaw,
                    cam_pitch,
                    cam_xyz_target,
                ) = self.get_cam_settings()

                self.physics_client = p.connect(p.GUI, options=opt_str)
                p.configureDebugVisualizer(
                    p.COV_ENABLE_GUI, 0, lightPosition=[-10, 0, 30]
                )  # lightPosition=[8, 0, 10])

                p.resetDebugVisualizerCamera(
                    cameraDistance=cam_distance,
                    cameraYaw=cam_yaw,
                    cameraPitch=cam_pitch,
                    cameraTargetPosition=cam_xyz_target,
                )
            else:
                self.physics_client = p.connect(p.DIRECT)
            if self.debug:
                self.debugger.setup()
            self.first_run_done = True

        else:
            p.resetSimulation()

        # setting up the physics client
        p.setGravity(self.gravity[0], self.gravity[1], self.gravity[2])
        p.setPhysicsEngineParameter(enableConeFriction=1)
        p.setRealTimeSimulation(
            0
        )  # only if this is set to 0 and the simulation is done with explicit steps will the torque control work correctly

        p.setTimeStep(self.bullet_time_step)

        # load the ground plane
        if self.ground_plane_height is not None:
            planeId = p.loadURDF(
                os.path.join(
                    os.path.dirname(__file__),
                    "general_definitions/plane_urdf/plane.urdf",
                ),
                basePosition=[0, 0, self.ground_plane_height],
                flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
            )
            p.changeDynamics(
                planeId, -1, lateralFriction=1
            )  # set ground plane friction

        for (
            manipulator,
            start_position,
            start_orientation,
            manipulator_base_constraint,
            manipulator_load_flag,
        ) in zip(
            self.manipulators,
            self.start_positions,
            self.start_orientations,
            self.manipulator_base_constraints,
            self.manipulator_load_flags,
        ):
            manipulator.load_to_pybullet(
                baseStartPos=start_position,
                baseStartOrn=start_orientation,
                baseConstraint=manipulator_base_constraint,
                physicsClient=self.physics_client,
                flags=manipulator_load_flag,
            )

        self.modify_physics()

        # run some steps with constant 0 actuation to have steady state at start
        start_action = np.zeros(self.action_space.shape)
        stabilization_time = 0.5  # time in the beginning to wait to stabilize # todo: should this maybe be given in the specific environment?
        self.applied_torque = start_action * self.torque_multiplier

        self.step_count = 0
        self.reward_component_info = deepcopy(self.reset_reward_component_info)
        for _ in range(int(np.ceil(stabilization_time / self.action_time))):
            self.stabilizing = True
            self.step(start_action)

        stabilized_obs = self.get_observation()
        self.stabilizing = False
        self.step_count = 0
        self.reward_component_info = deepcopy(self.reset_reward_component_info)
        self.ep_count += 1

        return stabilized_obs

    # Environments will automatically close() themselves when garbage is collected or when the program exits.
    def close(self):
        # disconnect pybullet
        p.disconnect()

        # delete urdf files
        for manipulator in self.manipulators:
            # todo: add try/except in case this urdf was already deleted
            if os.path.isfile(manipulator.manipulator_definition.urdf_filename):
                os.remove(manipulator.manipulator_definition.urdf_filename)

    def render(self, mode="human", close=False):
        # todo: we are currently using env.render wrong - fix or document carefully
        if mode == "human":
            True

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
            1.0,
            0.0,
            -80.0,
            [0.0, 0.00, 0.0],
        )

        return opt_str, cam_distance, cam_yaw, cam_pitch, cam_xyz_target
