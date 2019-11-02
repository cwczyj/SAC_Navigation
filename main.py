import torch
import glob
import os

from envs.myEnvs import make_vec_house3d_envs
from models.model import QNet,TanhGaussianPolicy, EvalPolicy
from collector.replaybuffer import ReplayBuffer
from algo.sac import SACTrainer
from algo.rl_algorithm import RLAlgorithm
from arguments import get_args


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


def experiment(args):
    print('Begin Training Networks')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_dir = os.path.expanduser(args.log_dir)
    cleanup_log_dir(log_dir)

    device = torch.device('cuda:'+str(args.device) if args.cuda else 'cpu')

    expl_envs = make_vec_house3d_envs(args.num_processes, log_dir, device, False)

    print('Initial Multi Processes Envs Success.')

    obs_dim = expl_envs.observation_space.low.size
    action_dim = expl_envs.action_space.low.size

    H = args.hidden_size
    recurrent_hidden_state_size = args.recurrent_hidden_state
    qf1 = QNet(
        action_size=action_dim,
        feature_size=args.obs_feature_size,
        output_size=1,
        hidden_sizes=[H, H],
        recurrent_hidden_size=recurrent_hidden_state_size,
    )
    qf2 = QNet(
        action_size=action_dim,
        feature_size=args.obs_feature_size,
        output_size=1,
        hidden_sizes=[H, H],
        recurrent_hidden_size=recurrent_hidden_state_size,
    )
    target_qf1 = QNet(
        action_size=action_dim,
        feature_size=args.obs_feature_size,
        output_size=1,
        hidden_sizes=[H, H],
        recurrent_hidden_size=recurrent_hidden_state_size,
    )
    target_qf2 = QNet(
        action_size=action_dim,
        feature_size=args.obs_feature_size,
        output_size=1,
        hidden_sizes=[H, H],
        recurrent_hidden_size=recurrent_hidden_state_size,
    )

    policy = TanhGaussianPolicy(
        hidden_sizes=[H, H],
        action_size=action_dim,
        obs_feature_size=args.obs_feature_size,
        recurrent_hidden_size=recurrent_hidden_state_size
    )

    eval_policy = EvalPolicy(policy)
    replay_buffer = ReplayBuffer(
        max_replay_buffer_size=int(args.max_replay_buffer_size / args.rnn_seq_len),
        observation_dim=obs_dim,
        action_dim=action_dim,
        rnn_seq_len=args.rnn_seq_len,
        agent_recurrent_hidden_states_size=recurrent_hidden_state_size,
        qf1_recurrent_hidden_states_size=recurrent_hidden_state_size,
        qf2_recurrent_hidden_states_size=recurrent_hidden_state_size,
    )

    sactrainer = SACTrainer(
        envs=expl_envs,
        policy=policy,
        eval_policy=eval_policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,

        policy_lr=args.policy_lr,
        qf_lr=args.qf_lr,

        device=device,
    )

    algorithm = RLAlgorithm(
        trainer=sactrainer,
        envs=expl_envs,
        replay_buffer=replay_buffer,
        batch_size=args.batch_size,
        max_path_length=args.max_train_path_len,
        rnn_seq_len=args.rnn_seq_len,
        num_epochs=args.num_train_epochs,
        num_eval_steps_per_epoch=args.eval_steps,
        num_expl_steps_per_train_loop=args.num_steps_per_loop,
        num_trains_per_train_loop=args.num_trains_per_loop,
        num_train_loops_per_epoch=args.num_train_loops_per_epoch,
        min_num_steps_before_training=args.befor_traning,
        save_path=str(args.save_path),
    )

    algorithm.to(device)
    algorithm.train()


if __name__ == '__main__':
    args = get_args()
    experiment(args)



