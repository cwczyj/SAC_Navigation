import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box
from torchvision import transforms

from baselines import bench
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.running_mean_std import RunningMeanStd

from envs.house3dEnv import House3DEnv
from models.model import ImageFeatureNet


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, {str(op)}, must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecNormalize(VecEnvWrapper):
    def __init__(self, venv, ob=True, ret=True, clipob=5., cliprew=5., ext_gamma=0.999, int_gamma=0.999, epsilon=1e-8):
        super(VecNormalize, self).__init__(venv)
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ext_ret_rms = RunningMeanStd(shape=()) if ret else None
        self.int_ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipobs = clipob
        self.cliprew = cliprew
        self.ext_ret = np.zeros(self.num_envs)
        self.ext_gamma = ext_gamma
        self.int_ret = np.zeros(self.num_envs)
        self.int_gamma = int_gamma
        self.epsilon = epsilon

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        return obs, rews, news, infos

    def reset(self):
        self.ext_ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return obs


class VecHouse3DEnv(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecHouse3DEnv, self).__init__(venv)
        self.device = device
        # TODO: Fix data types
        self.transform = ImageFeatureNet(
            checkpoint_path='models/model.pt'
        )
        self.trans = transforms.Compose([transforms.ToTensor()])

    def reset(self):
        obs = self.venv.reset()
        obs = self.transform_obs(obs)
        obs = obs.to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.cpu().numpy().astype(np.float)
        self.venv.step_async(actions)

    @property
    def h_info(self):
        return self.venv.get_attr('info')

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = self.transform_obs(obs)
        # obs = torch.from_numpy(obs).float().to(self.device)
        obs = obs.to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info

    def transform_obs(self, obs):
        assert len(obs.shape) == 4, "dims: process * H * W * C"
        s = obs.shape
        total_obs = torch.zeros(s[0], 6, s[1], s[2])
        for i in range(obs.shape[0]):
            # total_obs[i, :, :, :] = torch.from_numpy(obs[i].transpose(2, 0, 1))
            # total_obs[i] = total_obs[i] / 255.0
            rgb_img = obs[i, :, :, :3]
            dep_img = np.zeros(rgb_img.shape, dtype=rgb_img.dtype)
            for j in range(3):
                dep_img[..., j] = obs[i, :, :, 3]
            rgb_img = self.trans(rgb_img)
            dep_img = self.trans(dep_img)
            total_obs[i, :3] = rgb_img
            total_obs[i, 3:] = dep_img
        output_obs = self.transform(total_obs)
        return output_obs


def make_house3d_env(rank, log_dir, allow_early_resets):

    def _thunk():
        env = House3DEnv()

        if log_dir is not None:
            env = bench.Monitor(env,
                                os.path.join(log_dir, str(rank)),
                                allow_early_resets=allow_early_resets)
        return env

    return _thunk


def make_vec_house3d_envs(num_processes,
                          log_dir,
                          device,
                          allow_early_resets):
    envs = [
        make_house3d_env(i, log_dir, allow_early_resets)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = ShmemVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    envs.observation_shape = (3200,)
    envs.observation_space = Box(0, 1, shape=envs.observation_shape, dtype=np.uint8)

    envs = VecHouse3DEnv(envs, device)
    #envs = VecNormalize(envs)

    return envs


if __name__ == "__main__":
    device = torch.device('cuda:0')
    envs = make_vec_house3d_envs(2, '../logdir/', device, False)
    reset_obs = envs.reset()
    obs, rewards, dones, infos = envs.step(torch.Tensor([[0.2, -0.4], [0.2, -0.3]]))
    print('HHHHH.{}'.format(reset_obs))
