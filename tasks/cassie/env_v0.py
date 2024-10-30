from tasks.cassie.generator import FootTrajectoryGenerator

from mujoco_py import load_model_from_xml
from mujoco_py import MjViewer
from mujoco_py import MjSim
import mujoco_py

from scipy.spatial.transform import Rotation
from collections import OrderedDict
from collections import deque
from copy import deepcopy
import gymnasium as gym
import numpy as np
import xmltodict
import time
import sys
import os

ABS_PATH = os.path.dirname(os.path.abspath(__file__))


class Env(gym.Env):
    def __init__(
            self, use_fixed_base=False, 
            init_base_pos=[0.0, 0.0, 1.0], 
            init_base_quat=[1.0, 0.0, 0.0, 0.0],
            max_episode_length=1000, 
            is_earlystop=False,
            lin_vel_cmd_range=[-1.0, 1.0],
            ang_vel_cmd_range=[-0.5, 0.5]) -> None:

        # =========== for simulation parameter =========== #
        self.sim_dt = 0.002
        self.contro_freq = 50.0
        self.n_substeps = int(1/(self.sim_dt*self.contro_freq))
        self.env_dt = self.sim_dt*self.n_substeps
        self.use_fixed_base = use_fixed_base
        self.gravity = np.array([0, 0, -9.8])
        self.num_legs = 2

        # for init value
        self.init_base_pos = init_base_pos
        self.init_base_quat = init_base_quat

        # for Kp & Kd of actuator
        # order: abduct, thigh, knee
        self.Kps = np.array([100,  100,  88,  96,  50]*2) 
        self.Kds = np.array([10.0, 10.0, 8.0, 9.6, 5.0]*2)

        # joint limit
        self.lower_limits = np.array([-15, -22.5, -50, -164, -140]*2)*np.pi/180.0
        self.upper_limits = np.array([22.5, 22.5, 80, -37, -30]*2)*np.pi/180.0
        self.joint_names = [
            "left_hip_roll", "left_hip_yaw", "left_hip_pitch", "left_knee", "left_foot", 
            "right_hip_roll", "right_hip_yaw", "right_hip_pitch", "right_knee", "right_foot"
        ]

        # for mujoco object
        self.model = self._loadModel(use_fixed_base=self.use_fixed_base)
        self.sim = MjSim(self.model, nsubsteps=1)
        self.viewer = None

        # get sim id
        self.robot_id = self.sim.model.body_name2id('cassie_pelvis')
        self.geom_floor_id = self.sim.model.geom_name2id('floor')
        self.geom_foot_ids = [self.sim.model.geom_name2id(f'{name}_foot') for name in ['left', 'right']]

        # joint index offset
        if self.use_fixed_base:
            self.pos_idx_offset = 0
            self.vel_idx_offset = 0
        else:
            self.pos_idx_offset = 7
            self.vel_idx_offset = 6
        # ================================================ #

        # foot step trajectory generator
        self.generator = FootTrajectoryGenerator()
        self.generator.foot_height = 0.0
        self.nominal_joint_targets = self.generator.getJointTargets(0.0)
        nominal_lower_limits = self.lower_limits - self.nominal_joint_targets
        nominal_upper_limits = self.upper_limits - self.nominal_joint_targets
        nominal_action_bounds = np.max(np.abs(np.stack(
            [nominal_lower_limits, nominal_upper_limits], axis=0)), 
            axis=0).astype(np.float32)

        # environmental variables
        self.max_episode_length = max_episode_length
        self.is_earlystop = is_earlystop
        self.cur_step = 0
        self.num_history = 3
        self.joint_pos_history = deque(maxlen=self.num_history)
        self.joint_vel_history = deque(maxlen=self.num_history)
        self.joint_target_history = deque(maxlen=self.num_history)
        self.cmd_lin_vel = np.zeros(3)
        self.cmd_ang_vel = np.zeros(3)
        self.lin_vel_cmd_range = lin_vel_cmd_range
        self.ang_vel_cmd_range = ang_vel_cmd_range
        assert self.lin_vel_cmd_range[0] <= self.lin_vel_cmd_range[1]
        assert self.ang_vel_cmd_range[0] <= self.ang_vel_cmd_range[1]
        self.action = np.zeros_like(self.lower_limits)
        self.action_weight = 0.5
        self.power_consumption = 0.0

        # for gym environment
        self.state_keys = [
            'cmd_lin_vel', 'cmd_ang_vel', 'gravity_vector', 'base_lin_vel', 'base_ang_vel', 
            'joint_pos_list', 'joint_vel_list', 'phase_list', 
            'joint_pos_history', 'joint_vel_history', 'joint_target_history',
            'contact_list', 'base_height',
        ]
        state, info = self.reset()
        raw_state = self._getRawState()
        self.state_dim = state.shape[0]
        self.action_dim = len(self.lower_limits)
        self.reward_dim = len(self._getRewards(raw_state))
        self.cost_dim = len(self._getCosts(raw_state))
        self.observation_space = gym.spaces.Box(
            -np.inf*np.ones(self.state_dim, dtype=np.float32), 
            np.inf*np.ones(self.state_dim, dtype=np.float32), dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            -nominal_action_bounds, nominal_action_bounds, dtype=np.float32,
        )
        self.reward_space = gym.spaces.Box(
            -np.inf*np.ones(self.reward_dim, dtype=np.float32), 
            np.inf*np.ones(self.reward_dim, dtype=np.float32), dtype=np.float32,
        )
        self.cost_space = gym.spaces.Box(
            -np.inf*np.ones(self.cost_dim, dtype=np.float32), 
            np.inf*np.ones(self.cost_dim, dtype=np.float32), dtype=np.float32,
        )

    ################
    # public methods
    ################

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # reset sim & generator
        self.sim.reset()
        self.generator.reset()
        self.action = np.zeros_like(self.lower_limits)

        # reset joint pos & vel
        init_qpos_list = [
            -6.05507670e-04, -9.46848441e-04,  6.88587941e-01,  9.43366294e-01,
            2.79908818e-03,  2.57290345e-02, -3.30741919e-01, -1.46884023e+00,
            4.99414927e-03,  1.67887314e+00, 1.56343722e-03, -1.55491322e+00, 
            1.53641059e+00, -1.65963020e+00,  6.02820781e-04,  9.20671538e-04,
            6.88739578e-01,  9.43359484e-01, -2.79838679e-03, -2.57319768e-02,
            -3.30761119e-01, -1.46842817e+00,  4.47926175e-03,  1.67906987e+00,
            1.58396908e-03, -1.55491322e+00, 1.53641059e+00, -1.65963020e+00
        ]
        if not self.use_fixed_base:
            robot_pos = np.concatenate([self.init_base_pos, self.init_base_quat], axis=0)
            self.sim.data.qpos[:self.pos_idx_offset] = robot_pos
            self.sim.data.qvel[:self.vel_idx_offset] = 0.0
        self.sim.data.qpos[self.pos_idx_offset:] = init_qpos_list
        self.sim.data.qvel[self.vel_idx_offset:] = 0.0
        self.sim.forward()

        # simulate
        joint_targets = self.generator.getJointTargets(0.0)
        joint_targets = np.clip(joint_targets, self.lower_limits, self.upper_limits)
        for _ in range(self.num_history):
            for step_idx in range(self.n_substeps):
                p_error = joint_targets - self._getJointPosList()
                d_error = self._getJointVelList()
                torque = self.Kps*p_error - self.Kds*d_error
                torque[5:7] = -torque[5:7]
                self.sim.data.ctrl[:] = torque
                self.sim.step()
            # reset variables
            self.joint_pos_history.append(p_error)
            self.joint_vel_history.append(d_error)
            self.joint_target_history.append(joint_targets)

        # reset variables
        # self.pre_pos = deepcopy(self.sim.data.get_site_xpos('imu'))
        # self.pre_rot = deepcopy(self.sim.data.get_site_xmat('imu'))
        self.cur_step = 0
        self.is_terminated = False
        self.cmd_lin_vel = np.array([np.random.uniform(*self.lin_vel_cmd_range)] + [0.0 , 0.0])
        self.cmd_ang_vel = np.array([0.0, 0.0] + [np.random.uniform(*self.ang_vel_cmd_range)])

        # get state
        raw_state = self._getRawState()
        return self._convertState(raw_state), {}

    def setCommandVel(self, lin_vel, ang_vel):
        self.cmd_lin_vel = np.array([lin_vel, 0.0, 0.0])
        self.cmd_ang_vel = np.array([0.0, 0.0, ang_vel])
        raw_state = self._getRawState()
        return self._convertState(raw_state)

    def step(self, action):
        self.cur_step += 1
        if self.is_terminated:
            state = deepcopy(self.terminal_state)
            reward = deepcopy(self.terminal_reward)
            info = deepcopy(self.terminal_info)
        else:
            state, reward, terminate, truncate, info = self._step(action)
            if terminate:
                self.is_terminated = True
                self.terminal_state = deepcopy(state)
                self.terminal_reward = deepcopy(reward)
                self.terminal_info = deepcopy(info)
        terminate = False if not self.is_earlystop else self.is_terminated
        truncate = (self.cur_step >= self.max_episode_length)
        return state, reward, terminate, truncate, info

    def render(self, mode='human', size=(512, 512), **kwargs):
        if mode == 'rgb_array':
            # img = self.sim.render(*size, camera_name="frontview")
            # img = img[::-1,:,:]
            # img2 = self.sim.render(*size, camera_name="fixed")
            # img2 = img2[::-1,:,:]
            # img = np.concatenate([img, img2], axis=1)
            # return img
            img = self.sim.render(*size, camera_name="track")
            img = img[::-1,:,:]
            return img
        else:
            if self.viewer is None:
                self.viewer = MjViewer(self.sim)
                self._viewerSetup(self.viewer)
            self.viewer.render()
            # self._addMarker()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return

    #################
    # private methods
    #################

    def _loadModel(self, use_fixed_base=False):
        # load xml file
        robot_base_path = f"{ABS_PATH}/models/mjcf.xml"
        with open(robot_base_path) as f:
            robot_base_xml = f.read()
        xml = xmltodict.parse(robot_base_xml)
        body_xml = xml['mujoco']['worldbody']['body']
        if type(body_xml) not in [OrderedDict, dict]:
            body_xml = body_xml[0]

        # for mass
        def getMass(body_xml):
            mass = float(body_xml['inertial']['@mass'])
            if 'body' in body_xml.keys():
                if type(body_xml['body']) == list:
                    for sub_body in body_xml['body']:
                        mass += getMass(sub_body)
                else:
                    mass += getMass(body_xml['body'])
            return mass
        self.mass = getMass(body_xml)

        # for time interval
        if type(xml['mujoco']['option']) == list:
            for option in xml['mujoco']['option']:
                if '@gravity' in option.keys():
                    option['@gravity'] = ' '.join([f"{i}" for i in self.gravity])
                if '@timestep' in option.keys():
                    option['@timestep'] = self.sim_dt
        else:
            option = xml['mujoco']['option']
            option['@gravity'] = ' '.join([f"{i}" for i in self.gravity])
            option['@timestep'] = self.sim_dt

        # for base fix
        if use_fixed_base:
            del body_xml['joint']
        body_xml['@quat'] = ' '.join([f"{i}" for i in self.init_base_quat])
        body_xml['@pos'] = ' '.join([f"{i}" for i in self.init_base_pos])

        # convert xml to string & load model
        xml['mujoco']['compiler']['@meshdir'] = f'{ABS_PATH}/models/meshes'
        xml_string = xmltodict.unparse(xml)
        model = load_model_from_xml(xml_string)
        return model

    def _step(self, action):
        # ====== before simulation step ====== #
        joint_targets = self.generator.getJointTargets(self.cur_step*self.env_dt)
        joint_targets = np.clip(action + joint_targets, self.lower_limits, self.upper_limits)
        self.action = self.action*self.action_weight + joint_targets*(1.0 - self.action_weight)
        # ==================================== #

        # simulate
        self.power_consumption = 0.0
        for _ in range(self.n_substeps):
            p_error = self.action - self._getJointPosList()
            d_error = self._getJointVelList()
            torque = self.Kps*p_error - self.Kds*d_error
            torque[5:7] = -torque[5:7]
            self.sim.data.ctrl[:] = torque
            self.sim.step()
            self.power_consumption += np.mean(np.abs(
                self.sim.data.actuator_force*self._getJointVelList()))
        self.power_consumption /= self.n_substeps

        # ====== after simulation step ====== #
        self.joint_pos_history.append(p_error)
        self.joint_vel_history.append(d_error)
        self.joint_target_history.append(self.action)

        raw_state = self._getRawState()
        state = self._convertState(raw_state)
        rewards = self._getRewards(raw_state)
        costs = self._getCosts(raw_state)
        reward = np.concatenate([rewards, costs])

        body_angle = raw_state['gravity_vector'][2]/np.linalg.norm(raw_state['gravity_vector'])
        truncate = (self.cur_step >= self.max_episode_length)
        terminate = (body_angle >= 0)

        info = {}
        # =================================== #
        return state, reward, terminate, truncate, info

    def _getJointPosList(self):
        joint_pos_list = np.array([self.sim.data.get_joint_qpos(joint_name) for joint_name in self.joint_names])
        joint_pos_list[5:7] = -joint_pos_list[5:7]
        return joint_pos_list

    def _getJointVelList(self):
        joint_vel_list = np.array([self.sim.data.get_joint_qvel(joint_name) for joint_name in self.joint_names])            
        joint_vel_list[5:7] = -joint_vel_list[5:7]
        return joint_vel_list

    def _viewerSetup(self, viewer):
        viewer.cam.trackbodyid = self.robot_id
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90

    def _getRawState(self):
        state = {}
        state['cmd_lin_vel'] = self.cmd_lin_vel
        state['cmd_ang_vel'] = self.cmd_ang_vel

        base_pos = self.sim.data.get_site_xpos('imu')
        base_mat = self.sim.data.get_site_xmat('imu')
        yaw_angle = Rotation.from_matrix(base_mat).as_euler('zyx')[0]
        rot_mat = Rotation.from_rotvec([0.0, 0.0, yaw_angle]).as_matrix()
        state['base_height'] = base_pos[2:]

        gravity_vector = base_mat@self.gravity
        state['gravity_vector'] = rot_mat.T@gravity_vector

        base_lin_vel = rot_mat.T@self.sim.data.get_site_xvelp('imu')
        base_ang_vel = rot_mat.T@self.sim.data.get_site_xvelr('imu')
        state['base_lin_vel'] = base_lin_vel
        state['base_ang_vel'] = base_ang_vel

        state['joint_pos_list'] = self._getJointPosList()
        state['joint_vel_list'] = self._getJointVelList()

        state['phase_list'] = self.generator.getPhaseList()
        state['base_freq'] = np.array([self.generator.default_freq])
        state['freq_list'] = deepcopy(self.generator.freq_list)

        state['joint_pos_history'] = np.concatenate(list(self.joint_pos_history))
        state['joint_vel_history'] = np.concatenate(list(self.joint_vel_history))
        state['joint_target_history'] = np.concatenate(list(self.joint_target_history))

        contact_list = np.zeros(self.num_legs)
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            for geom_toe_idx, geom_toe_id in enumerate(self.geom_foot_ids):
                if contact.geom1 == geom_toe_id or contact.geom2 == geom_toe_id:
                    if contact.geom1 != contact.geom2:
                        contact_list[geom_toe_idx] = 1.0
        state['contact_list'] = contact_list

        collision = False
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            if contact.geom1 == contact.geom2:
                continue
            condition1 = (contact.geom1 == self.geom_floor_id) and (not contact.geom2 in self.geom_foot_ids)
            condition2 = (contact.geom2 == self.geom_floor_id) and (not contact.geom1 in self.geom_foot_ids)
            if condition1 or condition2:
                collision = True
                break
        state['collision'] = collision
        return state

    def _convertState(self, state):
        flatten_state = []
        for key in self.state_keys:
            flatten_state.append(state[key])
        state = np.concatenate(flatten_state)
        return np.array(state, dtype=np.float32)

    def _getRewards(self, state):
        ang_vel_error = (self.cmd_ang_vel[2] - state['base_ang_vel'][2])**2
        lin_vel_error = np.sum(np.square(state['base_lin_vel'][:2] - self.cmd_lin_vel[:2]))
        error = ang_vel_error + lin_vel_error
        power_reward = -1e-3*self.power_consumption
        reward = 0.1*(-error + power_reward)
        return np.array([reward])

    def _getCosts(self, state):
        costs = []

        # for body angle constraint
        a = -np.cos(15.0*(np.pi/180.0))
        x = state['gravity_vector'][2]/np.linalg.norm(state['gravity_vector'])
        costs.append(1.0 if x > a else 0.0)

        # for height
        a = 0.7
        x = state['base_height'][0]
        costs.append(1.0 if x < a else 0.0)

        # swing timing
        cost = 0.0
        for leg_idx in range(self.num_legs):
            cos_phase, sin_phase = state['phase_list'][2*leg_idx:2*(leg_idx+1)]
            if sin_phase < 0.0: # swing phase
                cost += 1.0 if state['contact_list'][leg_idx] else 0.0
            else: # stance phase
                cost += 0.0 if state['contact_list'][leg_idx] else 1.0
        cost /= self.num_legs
        costs.append(cost)
        return np.array(costs)

    def _addMarker(self):
        base_pos = self.sim.data.get_site_xpos('imu')
        base_mat = self.sim.data.get_site_xmat('imu')
        pos = base_pos + np.array([0.0, 0.0, 0.4])
        size = np.clip(self.cmd_ang_vel[2], -0.5, 0.5)/0.5
        self.viewer.add_marker(
            pos=pos,
            size=np.array([0.01, 0.01, size*0.5]),
            mat=np.eye(3),
            type=mujoco_py.const.GEOM_ARROW,
            rgba=np.array([1, 1, 0, 1]),
            label=''
        )
        size = np.clip(self.cmd_lin_vel[0], -0.5, 0.5)/0.5
        yaw_angle = Rotation.from_matrix(base_mat).as_euler('zyx')[0]
        mat = Rotation.from_rotvec([0.0, 0.0, yaw_angle]).as_matrix()@Rotation.from_rotvec([0.0, np.pi/2, 0.0]).as_matrix()
        self.viewer.add_marker(
            pos=pos,
            size=np.array([0.01, 0.01, size*0.5]),
            mat=mat,
            type=mujoco_py.const.GEOM_ARROW,
            rgba=np.array([0, 0, 1, 1]),
            label=''
        )
        sin_phase = self.generator.getPhaseList()[1]
        if sin_phase < 0.0:
            color = np.array([1, 0, 0, 1])
        else:
            color = np.array([0, 1, 0, 1])
        new_pos = pos + np.array([0.0, 0.0, 0.5])
        self.viewer.add_marker(
            pos=new_pos,
            size=np.ones(3)*0.1,
            mat=np.eye(3),
            type=mujoco_py.const.GEOM_SPHERE,
            rgba=color,
            label=''
        )


if __name__ == "__main__":
    # env = Env(use_fixed_base=True, init_base_pos=[0, 0, 1.0])
    env = Env(use_fixed_base=False)

    for i in range(10):
        env.reset()
        start_t = time.time()
        global_t = 0.0
        elapsed_t = 0.0
        action = np.zeros(env.action_space.shape[0])
        # action[0] = action[5+0] = 0.5
        for i in range(100):
            # action = env.action_space.sample()
            # s, r, d, info = env.step(action*0.5)
            s, r, d, info = env.step(action)
            env.render()
            global_t += env.env_dt

            elapsed_t = time.time() - start_t
            if elapsed_t < global_t:
                time.sleep(global_t - elapsed_t)

            if d: break
        # print(env.sim.data.qpos)
        # exit()

    #         sys.stdout.write("\rsim time : {:.3f} s, real time : {:.3f} s".format(global_t, elapsed_t))
    #         sys.stdout.flush()
    # sys.stdout.write(f"\rgoodbye.{' '*50}\n")
