import numpy as np
import torch.nn as nn

class ReplayBuffer(nn.Module):

    def __init__(self,
                 max_replay_buffer_size,
                 observation_dim,
                 action_dim,
                 rnn_seq_len,
                 agent_recurrent_hidden_states_size,
                 qf1_recurrent_hidden_states_size,
                 qf2_recurrent_hidden_states_size,
                 ):

        super(ReplayBuffer, self).__init__()
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._rnn_seq_len = rnn_seq_len
        self._agent_recurrent_hidden_states = agent_recurrent_hidden_states_size
        self._qf1_recurrent_hidden_states = qf1_recurrent_hidden_states_size
        self._qf2_recurrent_hidden_states = qf2_recurrent_hidden_states_size
        self._max_replay_buffer_size = max_replay_buffer_size

        self._observations = np.zeros((rnn_seq_len, max_replay_buffer_size, observation_dim))
        self._next_observations = np.zeros((rnn_seq_len, max_replay_buffer_size, observation_dim))
        self._actions = np.zeros((rnn_seq_len, max_replay_buffer_size, action_dim))
        self._rewards = np.zeros((rnn_seq_len, max_replay_buffer_size, 1))
        self._terminals = np.zeros((rnn_seq_len, max_replay_buffer_size, 1))

        self._agent_recurrent_hidden_states = np.zeros((max_replay_buffer_size,
                                                        agent_recurrent_hidden_states_size))
        self._qf1_recurrent_hidden_states = np.zeros((max_replay_buffer_size,
                                                      qf1_recurrent_hidden_states_size))
        self._qf2_recurrent_hidden_states = np.zeros((max_replay_buffer_size,
                                                      qf2_recurrent_hidden_states_size))

        self._next_agent_recurrent_hidden_states = np.zeros((max_replay_buffer_size,
                                                             agent_recurrent_hidden_states_size))
        self._next_qf1_recurrent_hidden_states = np.zeros((max_replay_buffer_size,
                                                           qf1_recurrent_hidden_states_size))
        self._next_qf2_recurrent_hidden_states = np.zeros((max_replay_buffer_size,
                                                           qf2_recurrent_hidden_states_size))

        # seem no need save env infos
        self._env_infos = None

        self._top = 0
        self._size = 0

    def add_sample(self, observation, next_observation,
                   action, reward, terminal, agent_recurrent_hidden_state,
                   qf1_recurrent_hidden_state, qf2_recurrent_hidden_state,
                   next_agent_recurrent_hidden_state,
                   next_qf1_recurrent_hidden_state, next_qf2_recurrent_hidden_state,
                   ):

        self._observations[:, self._top, :] = observation
        self._actions[:, self._top, :] = action
        self._rewards[:, self._top, :] = reward
        self._terminals[:, self._top, :] = terminal
        self._next_observations[:, self._top, :] = next_observation

        self._agent_recurrent_hidden_states[self._top, :] = agent_recurrent_hidden_state
        self._qf1_recurrent_hidden_states[self._top, :] = qf1_recurrent_hidden_state
        self._qf2_recurrent_hidden_states[self._top, :] = qf2_recurrent_hidden_state

        self._next_agent_recurrent_hidden_states[self._top, :] = next_agent_recurrent_hidden_state
        self._next_qf1_recurrent_hidden_states[self._top, :] = next_qf1_recurrent_hidden_state
        self._next_qf2_recurrent_hidden_states[self._top, :] = next_qf2_recurrent_hidden_state

        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        new_batch_size = self._rnn_seq_len * batch_size
        batch = dict(
            observations=self._observations[:, indices, :].reshape(new_batch_size, -1),
            actions=self._actions[:, indices, :].reshape(new_batch_size, -1),
            rewards=self._rewards[:, indices, :].reshape(new_batch_size, -1),
            terminals=self._terminals[:, indices, :].reshape(new_batch_size, -1),
            next_observations=self._next_observations[:, indices, :].reshape(new_batch_size, -1),

            agent_recurrent_hidden_states=self._agent_recurrent_hidden_states[indices, :].reshape(batch_size, -1),
            qf1_recurrent_hidden_states=self._qf1_recurrent_hidden_states[indices, :].reshape(batch_size, -1),
            qf2_recurrent_hidden_states=self._qf2_recurrent_hidden_states[indices, :].reshape(batch_size, -1),

            next_agent_recurrent_hidden_states=self._next_agent_recurrent_hidden_states[indices, :].reshape(batch_size, -1),
            next_qf1_recurrent_hidden_states=self._next_qf1_recurrent_hidden_states[indices, :].reshape(batch_size, -1),
            next_qf2_recurrent_hidden_states=self._next_qf2_recurrent_hidden_states[indices, :].reshape(batch_size, -1),
        )

        return batch

    def num_steps_can_sample(self):
        return self._size

    def add_trajectory(self, trajectory):

        for i in range(trajectory['observations'].shape[1]):
            self.add_sample(
                observation=trajectory['observations'][:, i, :],
                next_observation=trajectory['next_observations'][:, i, :],
                action=trajectory['actions'][:, i, :],
                reward=trajectory['rewards'][:, i, :],
                terminal=trajectory['terminals'][:, i, :],

                agent_recurrent_hidden_state=trajectory['agent_recurrent_hidden_states'][i],
                qf1_recurrent_hidden_state=trajectory['qf1_recurrent_hidden_states'][i],
                qf2_recurrent_hidden_state=trajectory['qf2_recurrent_hidden_states'][i],

                next_agent_recurrent_hidden_state=trajectory['next_agent_recurrent_hidden_states'][i],
                next_qf1_recurrent_hidden_state=trajectory['next_qf1_recurrent_hidden_states'][i],
                next_qf2_recurrent_hidden_state=trajectory['next_qf2_recurrent_hidden_states'][i],
            )


