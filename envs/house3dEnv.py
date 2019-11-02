import os

import cv2
import numpy as np
from House3D import objrender, Environment, load_config

from envs.house3d_config import get_configs
from envs import seeding
from envs.gen_point_cloud import gen_point_cloud
import gym
from gym import spaces

RENDERING_GPU = 0
CUDA_DEVICE = os.environ.get('CUDA_VISIBLE_DEVICES')
if CUDA_DEVICE:
    RENDERING_GPU = 0
HEIGHT_THRESHOLD = (0.2, 1.8)

# Fwd, Bck, L, R, Lrot, Rrot
# discrete_actions = [(1., 0., 0.), (-1., 0., 0.), (0., -1., 0.),
#                     (0., 1., 0.), (0., 0., -1.), (0., 0., 1.)]

discrete_actions = [(1., 0., 0.), (0., 0., -1.), (0., 0., 1.)]
n_discrete_actions = len(discrete_actions)


class House3DEnv(gym.Env):
    def __init__(self,
                 train_mode=True,
                 area_reward_scale=0.0005,
                 collision_penalty=0.01,
                 step_penalty=0.0,
                 max_depth=2.0,
                 render_door=False,
                 start_indoor=True,
                 ignore_collision=False,
                 ob_dilation_kernel=5,
                 depth_signal=True,
                 max_steps=500):
        self.seed()
        self.configs = get_configs()
        self.env = None

        self.train_mode = train_mode
        self.render_door = render_door
        self.ignore_collision = ignore_collision
        self.start_indoor = start_indoor
        self.render_height = self.configs['render_height']
        self.render_width = self.configs['render_width']
        self.ob_dilation_kernel = ob_dilation_kernel
        self.config = load_config(self.configs['path'],
                                  prefix=self.configs['par_path'])
        self.move_sensitivity = self.configs['move_sensitivity']
        self.rot_sensitivity = self.configs['rot_sensitivity']
        self.train_houses = self.configs['train_houses']
        self.test_houses = self.configs['test_houses']

        if train_mode:
            self.houses_id = self.train_houses
        else:
            self.houses_id = self.test_houses
        self.depth_threshold = (0, max_depth)
        self.area_reward_scale = area_reward_scale
        self.collision_penalty = collision_penalty
        self.step_penalty = step_penalty
        self.max_step = max_steps

        n_channel = 3
        if depth_signal:
            n_channel += 1

        self.observation_shape = (self.render_width, self.render_height, n_channel)
        self.observation_space = spaces.Box(0, 255, shape=self.observation_shape, dtype=np.uint8)

        #self.action_space = spaces.Discrete(n_discrete_actions)
        self.action_space = spaces.Box(low=np.array([0.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)

        self.tracker = []
        self.num_rotate = 0
        self.right_rotate = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random()
        return [seed]

    def constrain_to_pm_pi(self, theta):
        return (theta + 180) % 360 - 180

    def get_camera_grid_pos(self):
        current_pos = np.array([self.env.cam.pos.x,
                                self.env.cam.pos.z,
                                self.constrain_to_pm_pi(self.env.cam.yaw)])
        grid_pos = np.array(self.env.house.to_grid(current_pos[0], current_pos[1]))
        return current_pos, grid_pos

    def get_obs(self):
        self.env.set_render_mode('rgb')
        rgb = self.env.render()
        self.env.set_render_mode('depth')
        depth = self.env.render()
        infmask = depth[:, :, 1]
        depth = depth[:, :, 0] * (infmask == 0)
        true_depth = depth.astype(np.float32) / 255.0 * 20.0
        extrinsics = self.env.cam.getExtrinsicsNumpy()
        return rgb, np.expand_dims(depth, -1), true_depth, extrinsics

    def safe_assign(self, im_map, x_idx, y_idx, value):
        try:
            im_map[x_idx, y_idx] = value
        except IndexError:
            valid_idx1 = np.logical_and(x_idx >= 0, x_idx < im_map.shape[0])
            valid_idx2 = np.logical_and(y_idx >= 0, y_idx < im_map.shape[1])
            valid_idx = np.logical_and(valid_idx1, valid_idx2)
            im_map[x_idx[valid_idx], y_idx[valid_idx]] = value

    def get_seen_area(self, rgb, depth, extrinsics, out_mat, inv_E=True):
        points, points_colors = gen_point_cloud(depth, rgb, extrinsics,
                                                depth_threshold=self.depth_threshold,
                                                inv_E=inv_E)
        grid_locs = np.floor((points[:, [0, 2]] - self.L_min) / self.grid_size).astype(int)
        grids_mat = np.zeros((self.grids_mat.shape[0],
                              self.grids_mat.shape[1]),
                              dtype=np.uint8)

        high_filter_idx = points[:, 1] < HEIGHT_THRESHOLD[1]
        low_filter_idx = points[:, 1] > HEIGHT_THRESHOLD[0]
        obstacle_idx = np.logical_and(high_filter_idx, low_filter_idx)

        self.safe_assign(grids_mat, grid_locs[high_filter_idx, 0],
                         grid_locs[high_filter_idx, 1], 2)
        kernel = np.ones((3, 3), np.uint8)
        grids_mat = cv2.morphologyEx(grids_mat, cv2.MORPH_CLOSE, kernel)

        obs_mat = np.zeros((self.grids_mat.shape[0], self.grids_mat.shape[1]), dtype=np.uint8)
        self.safe_assign(obs_mat, grid_locs[obstacle_idx, 0],
                         grid_locs[obstacle_idx, 1], 1)
        kernel = np.ones((self.ob_dilation_kernel, self.ob_dilation_kernel), np.uint8)
        obs_mat = cv2.morphologyEx(obs_mat, cv2.MORPH_CLOSE, kernel)
        obs_idx = np.where(obs_mat == 1)
        self.safe_assign(grids_mat, obs_idx[0], obs_idx[1], 1)

        out_mat[np.where(grids_mat == 2)] = 2
        out_mat[np.where(grids_mat == 1)] = 1

        seen_area = np.sum(out_mat > 0)
        #cal_seen_area = np.sum(out_mat == 1)
        return seen_area

    def cal_reward(self, rgb, depth, extrinsics, collision_flag):
        if collision_flag:
            reward = -1.0 * self.collision_penalty
            area_reward = 0.0
            filled_grid_num = self.seen_area
        else:
            filled_grid_num = self.get_seen_area(rgb, depth,
                                                 extrinsics,
                                                 self.grids_mat,
                                                 inv_E=True)
            area_reward = (filled_grid_num - self.seen_area)
            reward = area_reward * self.area_reward_scale
            reward -= self.step_penalty
        self.seen_area = filled_grid_num
        raw_reward = {'area': area_reward, 'collision_flag': collision_flag}
        return reward, filled_grid_num, raw_reward

    def reset(self, house_id=None, x=None, y=None, yaw=None):
        if not self.train_mode:
            obs_map = self.env.house.obsMap.T
            self.obs_pos = obs_map == 1
            self.traj = []
            self.traj_actions = []
            self.grid_traj = []

        if house_id is None:
            house_id = self.np_random.choice(self.houses_id, 1)[0]
        self.hid = house_id
        if self.env is not None:
            del self.api
            del self.env
        self.api = objrender.RenderAPI(self.render_width,
                                       self.render_height,
                                       device=RENDERING_GPU)

        self.env = Environment(self.api, house_id, self.config,
                               GridDet=self.configs['GridDet'],
                               RenderDoor=self.render_door,
                               StartIndoor=self.start_indoor)

        self.tracker = []
        self.num_rotate = 0
        self.right_rotate = 0
        self.L_min = self.env.house.L_lo
        self.L_max = self.env.house.L_hi
        self.grid_size = self.env.house.grid_det
        grid_num = np.array([self.env.house.n_row[0] + 1,
                             self.env.house.n_row[1] + 1])
        self.grids_mat = np.zeros(tuple(grid_num), dtype=np.uint8)
        self.max_grid_size = np.max(grid_num)
        self.max_seen_area = float(np.prod(grid_num))
        self.env.reset(x=x, y=y, yaw=yaw)
        self.start_pos, self.grid_start_pos = self.get_camera_grid_pos()
        if not self.train_mode:
            self.traj.append(self.start_pos.tolist())
            self.grid_traj.append(self.grid_start_pos.tolist())

        rgb, depth, true_depth, extrinsics = self.get_obs()
        self.seen_area = self.get_seen_area(rgb, true_depth, extrinsics, self.grids_mat)

        self.ep_len = 0
        self.ep_reward = 0
        self.collision_times = 0
        ret_obs = np.concatenate((rgb, depth), axis=-1)
        return ret_obs

    def motion_primitive(self, action):
        collision_flag = False
        det_fwd = np.clip(action[0] + 0.8, 0.0, 2.0) / 2.0
        tmp_alpha = 0.0
        if action[2] > 0:
            tmp_alpha = 1.0
        elif action[2] < 0:
            tmp_alpha = -1.0
        det_rot = np.clip(action[1] + 0.8, 0.0, 1.6) * 0.5 * tmp_alpha
        move_fwd = det_fwd * self.move_sensitivity
        rotation = det_rot * self.rot_sensitivity

        if not self.env.move_forward(move_fwd, 0.0):
            collision_flag = True
        else:
            self.env.rotate(rotation)

        return collision_flag

    def step(self, action):
        collision_flag = self.motion_primitive(action)

        rgb, depth, true_depth, extrinsics = self.get_obs()

        current_pos, grid_current_pos = self.get_camera_grid_pos()
        if not self.train_mode:
            self.traj_actions.append(int(action))
            self.traj.append(current_pos.tolist())
            self.grid_traj.append(grid_current_pos.tolist())

        self.tracker.append([grid_current_pos[0], grid_current_pos[1],
                             self.env.cam.front.x, self.env.cam.front.z,
                             self.env.cam.right.x, self.env.cam.right.z])

        if action[1] > -0.8 and action[2] > 0:
            self.right_rotate += 1

        if action[1] > -0.8 and action[2] != 0:
            self.num_rotate += 1

        reward, seen_area, raw_reward = self.cal_reward(rgb,
                                                        true_depth,
                                                        extrinsics,
                                                        collision_flag)

        self.ep_len += 1
        self.ep_reward += reward
        if collision_flag:
            self.collision_times += 1
        info = {'reward_so_far': self.ep_reward, 'steps_so_far': self.ep_len,
                'seen_area': seen_area, 'collisions': self.collision_times,
                'start_pose': self.start_pos, 'house_id': self.hid,
                'collision_flag': collision_flag, 'grid_current_pos': grid_current_pos - self.grid_start_pos,
                'current_rot': current_pos[2] - self.start_pos[2]}
        info = {**info, **raw_reward, **self.info}

        if self.ep_len >= self.max_step:
            done = True
            info.update({'bad_transition': True})
        else:
            done = False
        ret_obs = np.concatenate((rgb, depth), axis=-1)

        return ret_obs, reward, done, info

    @property
    def house(self):
        return self.env.house

    @property
    def info(self):
        ret = self.env.info
        ret['track'] = self.tracker
        ret['total_rotate'] = self.num_rotate
        ret['right_rotate'] = self.right_rotate
        if not self.train_mode:
            ret['traj_actions'] = self.traj_actions
            ret['traj'] = self.traj
            ret['grid_traj'] = self.grid_traj
        return ret

if __name__ == "__main__":
    env = House3DEnv()

    reset_obs = env.reset()
    while True:
        rgb = reset_obs[:, :, :3]
        cv2.imshow("rgb_show", rgb[:, :, ::-1])

        key = cv2.waitKey(0)
        if key == 27 or key == ord('q'):  # esc
            reset_obs = env.reset()
        elif key == ord('w'):
            reset_obs, _, _, _ = env.step(0)
        elif key == ord('a'):
            reset_obs, _, _, _ = env.step(1)
        elif key == ord('d'):
            reset_obs, _, _, _ = env.step(2)
