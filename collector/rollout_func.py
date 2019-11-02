import torch
import numpy as np
import os
from collections import deque

def rollout(
        envs,
        agent,
        qf1,
        qf2,
        max_trajectory_len=500,
        rnn_seq_len=20
):
    """
    need save trajectory as : max_trajectory_len / rnn_seq_len elements;
    every elements contains:
        - rnn_seq_len observations
        - actions
        - rewards
        - next_observation
        - one recurrent_hidden_state

    """

    observations = []
    next_observations = []
    actions = []
    rewards = []
    terminals = []
    agent_recurrent_hidden_states = []
    qf1_recurrent_hidden_states = []
    qf2_recurrent_hidden_states = []

    next_agent_recurrent_hidden_states = []
    next_qf1_recurrent_hidden_states = []
    next_qf2_recurrent_hidden_states = []

    agent_recurrent_hidden_state = []
    qf1_recurrent_hidden_state = []
    qf2_recurrent_hidden_state = []

    obs = envs.reset()
    agent_state = torch.zeros(size=(obs.size(0), agent.recurrent_hidden_size),
                              device=obs.device)
    qf1_state = torch.zeros(size=(obs.size(0), qf1.recurrent_hidden_size),
                            device=obs.device)
    qf2_state = torch.zeros(size=(obs.size(0), qf2.recurrent_hidden_size),
                            device=obs.device)
    agent_recurrent_hidden_state.append(agent_state.to('cpu').numpy())
    qf1_recurrent_hidden_state.append(qf1_state.to('cpu').numpy())
    qf2_recurrent_hidden_state.append(qf2_state.to('cpu').numpy())

    observation = []
    observation.append(obs.to('cpu').numpy())
    action = []
    reward = []
    terminal = []
    for i in range(0, max_trajectory_len, rnn_seq_len):
        for j in range(rnn_seq_len):
            with torch.no_grad():
                output = agent(obs, agent_state)
            a = output[0]
            agent_state = output[4]
            action.append(a.to('cpu').numpy())
            agent_recurrent_hidden_state.append(agent_state.to('cpu').numpy())

            with torch.no_grad():
                _, qf1_state = qf1(obs, a, qf1_state)
                _, qf2_state = qf2(obs, a, qf2_state)
            qf1_recurrent_hidden_state.append(qf1_state.to('cpu').numpy())
            qf2_recurrent_hidden_state.append(qf2_state.to('cpu').numpy())

            obs, r, d, _ = envs.step(a)
            observation.append(obs.to('cpu').numpy())
            reward.append(r.to('cpu').numpy())
            terminal.append(np.asarray(d, dtype=np.float32).reshape(-1, 1))

        observations.append(np.asarray(observation[:-1]))
        next_observations.append(np.asarray(observation[1:]))
        actions.append(np.asarray(action))
        rewards.append(np.asarray(reward))
        terminals.append(np.asarray(terminal))
        agent_recurrent_hidden_states.append(agent_recurrent_hidden_state[0])
        qf1_recurrent_hidden_states.append(qf1_recurrent_hidden_state[0])
        qf2_recurrent_hidden_states.append(qf2_recurrent_hidden_state[0])

        next_agent_recurrent_hidden_states.append(agent_recurrent_hidden_state[1])
        next_qf1_recurrent_hidden_states.append(qf1_recurrent_hidden_state[1])
        next_qf2_recurrent_hidden_states.append(qf2_recurrent_hidden_state[1])

        observation = [observation[-1]]
        action.clear()
        reward.clear()
        terminal.clear()
        agent_recurrent_hidden_state = [agent_recurrent_hidden_state[-1]]
        qf1_recurrent_hidden_state = [qf1_recurrent_hidden_state[-1]]
        qf2_recurrent_hidden_state = [qf2_recurrent_hidden_state[-1]]



    observations = np.concatenate(observations, axis=1)
    next_observations = np.concatenate(next_observations, axis=1)
    actions = np.concatenate(actions, axis=1)
    rewards = np.concatenate(rewards, axis=1)
    terminals = np.concatenate(terminals, axis=1)
    agent_recurrent_hidden_states = np.concatenate(agent_recurrent_hidden_states, axis=0)
    qf1_recurrent_hidden_states = np.concatenate(qf1_recurrent_hidden_states, axis=0)
    qf2_recurrent_hidden_states = np.concatenate(qf2_recurrent_hidden_states, axis=0)

    next_agent_recurrent_hidden_states = np.concatenate(next_agent_recurrent_hidden_states, axis=0)
    next_qf1_recurrent_hidden_states = np.concatenate(next_qf1_recurrent_hidden_states, axis=0)
    next_qf2_recurrent_hidden_states = np.concatenate(next_qf2_recurrent_hidden_states, axis=0)

    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        next_observations=next_observations,

        agent_recurrent_hidden_states=agent_recurrent_hidden_states,
        qf1_recurrent_hidden_states=qf1_recurrent_hidden_states,
        qf2_recurrent_hidden_states=qf2_recurrent_hidden_states,

        next_agent_recurrent_hidden_states=next_agent_recurrent_hidden_states,
        next_qf1_recurrent_hidden_states=next_qf1_recurrent_hidden_states,
        next_qf2_recurrent_hidden_states=next_qf2_recurrent_hidden_states,
    )


def write_tracker(eval_num, i, tracker, sub=None):
    output_dir = './tracker_data/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if sub is None:
        fname = os.path.join(output_dir, 'track_'+str(eval_num%100)+'_'+str(i).zfill(4)+'.dat')
    else:
        fname = os.path.join(output_dir, 'track_'+sub+'_'+str(i).zfill(4)+'.dat')
    with open(fname, 'w') as f:
        for t in tracker:
            f.write('{} {} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(t[0],
                t[1], t[2], t[3], t[4], t[5]))


def eval_rollout(
        envs,
        eval_agent,
        eval_num,
        max_trajectory_len=500,
):
    episode_rewards = deque()
    episode_seen_area = deque()
    episode_total_rotate = deque()
    episode_total_right_rotate = deque()
    writer_counter = 0

    obs = envs.reset()
    recurrent_hidden_state = torch.zeros(size=(obs.size(0), eval_agent.recurrent_hidden_size),
                                         device=obs.device)

    for i in range(max_trajectory_len):
        with torch.no_grad():
            action, recurrent_hidden_state = eval_agent(obs, recurrent_hidden_state)

        obs, r, d, infos = envs.step(action)

        for info in infos:
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])
                episode_seen_area.append(info['seen_area'])
                episode_total_rotate.append(info['total_rotate'])
                episode_total_right_rotate.append(info['right_rotate'])
        # for i, done in enumerate(d):
        #     if done:
        #         write_tracker(eval_num, writer_counter, infos[i]['track'])
        #         writer_counter += 1

    return episode_rewards, episode_seen_area, episode_total_rotate, episode_total_right_rotate

