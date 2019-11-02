import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn


class SACTrainer(object):
    def __init__(self,
                 envs,
                 policy,
                 eval_policy,
                 qf1,
                 qf2,
                 target_qf1,
                 target_qf2,

                 discount=0.99,
                 reward_scale=10.0,

                 policy_lr=5e-4,
                 qf_lr=1e-3,
                 optimizer_class=optim.Adam,

                 soft_target_tau=1e-2,
                 target_update_period=10,
                 plotter=None,

                 use_automatic_entropy_tuning=True,
                 target_entropy=None,
                 device=None,
                 ):
        super(SACTrainer, self).__init__()

        self.envs = envs
        self.policy = policy
        self.eval_policy = eval_policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.device = device

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.envs.action_space.shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optim = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )

        self.qf1_optim = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )

        self.qf2_optim = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self._n_train_steps_total = 1
        self._num_train_steps = 0

    def soft_update_from_to(self, qf, target_qf, soft_target_tau):
        for target_param, param in zip(target_qf.parameters(), qf.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_target_tau) + param.data * soft_target_tau
            )

    def train_from_torch(self, batch):
        obs = batch['observations']
        actions = batch['actions']
        terminals = batch['terminals']
        next_obs = batch['next_observations']
        rewards = batch['rewards']
        agent_recurrent_hidden_states = batch['agent_recurrent_hidden_states']
        qf1_recurrent_hidden_states = batch['qf1_recurrent_hidden_states']
        qf2_recurrent_hidden_states = batch['qf2_recurrent_hidden_states']

        next_agent_recurrent_hidden_states = batch['next_agent_recurrent_hidden_states']
        next_qf1_recurrent_hidden_states = batch['next_qf1_recurrent_hidden_states']
        next_qf2_recurrent_hidden_states = batch['next_qf2_recurrent_hidden_states']

        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, agent_recurrent_hidden_states, reparameterize=True, return_log_prob=True
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions, qf1_recurrent_hidden_states)[0],
            self.qf2(obs, new_obs_actions, qf2_recurrent_hidden_states)[0],
        )

        policy_loss = (alpha * log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        q1_pred, _ = self.qf1(obs, actions, qf1_recurrent_hidden_states)
        q2_pred, _ = self.qf2(obs, actions, qf2_recurrent_hidden_states)

        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, next_agent_recurrent_hidden_states, reparameterize=True, return_log_prob=True,
        )

        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions, next_qf1_recurrent_hidden_states)[0],
            self.target_qf2(next_obs, new_next_actions, next_qf2_recurrent_hidden_states)[0],
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Update networks
        """
        self.qf1_optim.zero_grad()
        qf1_loss.backward()
        self.qf1_optim.step()

        self.qf2_optim.zero_grad()
        qf2_loss.backward()
        self.qf2_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            self.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
            self.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)
            self._n_train_steps_total = 0

        self._n_train_steps_total += 1

    @property
    def networks(self):
        return dict(
            policy=self.policy,
            eval_policy=self.eval_policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
        )

    def np_to_pytorch_batch(self, np_batch):
        tmp_dict = {}
        for k, v in np_batch.items():
            converted_value = torch.from_numpy(v).to(self.device).float()
            tmp_dict[k]=converted_value

        return tmp_dict

    def train(self, np_batch):
        self._num_train_steps += 1
        batch = self.np_to_pytorch_batch(np_batch)
        self.train_from_torch(batch)

