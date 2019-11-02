from collector.rollout_func import rollout, eval_rollout
import gtimer as gt
from tensorboardX import SummaryWriter
import numpy as np
import torch
import os
from collections import OrderedDict
import datetime

class RLAlgorithm(object):
    def __init__(
            self,
            trainer,
            envs,
            replay_buffer,
            batch_size,
            max_path_length,
            rnn_seq_len,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch,
            min_num_steps_before_training,
            save_path,
    ):
        self.trainer = trainer
        self.expl_env = envs
        self.eval_env = envs
        self.replay_buffer = replay_buffer
        self._start_epoch = 0
        self._save_path = save_path

        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.rnn_seq_len = rnn_seq_len
        self.num_epochs = int(num_epochs)
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training

        self.writer = SummaryWriter()

    def to(self, device):
        for k, v in self.trainer.networks.items():
            v.to(device)

    def training_mode(self, mode):
        for k, v in self.trainer.networks.items():
            if mode:
                v.train()
            else:
                v.eval()

    def train(self):
        if self.min_num_steps_before_training > 0:
            for _ in range(0, self.min_num_steps_before_training, self.max_path_length):
                patch_trajectory = rollout(self.expl_env, self.trainer.policy,
                                           self.trainer.qf1, self.trainer.qf2,
                                           self.max_path_length, self.rnn_seq_len)
                self.replay_buffer.add_trajectory(patch_trajectory)

        for epoch in gt.timed_for(
            range(self._start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            rewards, seen_area, total_rotate, right_rotate = eval_rollout(self.eval_env, self.trainer.eval_policy,
                                                                          epoch, self.num_eval_steps_per_epoch)
            self.writer.add_scalar('eval/mean_reward', np.mean(rewards), epoch)
            self.writer.add_scalar('eval/mean_sean_area', np.mean(seen_area), epoch)
            self.writer.add_scalar('eval/max_reward', np.max(rewards), epoch)
            self.writer.add_scalar('eval/max_sean_area', np.max(seen_area), epoch)
            self.writer.add_scalar('eval/min_reward', np.min(rewards), epoch)
            self.writer.add_scalar('eval/min_sean_area', np.min(seen_area), epoch)
            self.writer.add_scalar('eval/mean_rotate_ratio', abs(0.5 - np.sum(right_rotate) / np.sum(total_rotate)), epoch)

            gt.stamp('evalution_sampling', unique=False)

            for _ in range(self.num_train_loops_per_epoch):
                for _ in range(0, self.num_expl_steps_per_train_loop, self.max_path_length):
                    patch_trajectory = rollout(self.expl_env, self.trainer.policy,
                                               self.trainer.qf1, self.trainer.qf2,
                                               self.max_path_length, self.rnn_seq_len)
                    gt.stamp('exploration sampling', unique=False)

                    self.replay_buffer.add_trajectory(patch_trajectory)
                    gt.stamp('data storing', unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    train_batch_data = self.replay_buffer.random_batch(
                        self.batch_size
                    )
                    self.trainer.train(train_batch_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)

            self._end_epoch()

    def _get_epoch_timings(self):
        times_itrs = gt.get_times().stamps.itrs
        times = OrderedDict()
        epoch_time = 0
        for key in sorted(times_itrs):
            time = times_itrs[key][-1]
            epoch_time += time
            times['time/{} (s)'.format(key)] = time
        times['time/epoch (s)'] = epoch_time
        times['time/total (s)'] = gt.get_times().total
        return times

    def _end_epoch(self):
        for k, v in self.trainer.networks.items():
            torch.save(
                v,
                os.path.join(self._save_path, str(k) + '.pt')
            )

        gt.stamp('saving')

        times = self._get_epoch_timings()

        for k, v in times.items():
            print('{} ... {} \n'.format(k, str(datetime.timedelta(seconds=v))))
