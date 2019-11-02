import argparse

import torch

def get_args():
    parser = argparse.ArgumentParser(description='SAC')
    parser.add_argument(
       '--seed',
        type=int,
        default=1,
        help='random seed (default: 1)')
    parser.add_argument(
        '--log-dir',
        default='./logdir',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--num-processes',
        type=int,
        default=8,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--device',
        type=int,
        default=1,
        help='Cuda device used for training networks'
    )
    parser.add_argument(
        '--obs-feature-size',
        type=int,
        default=512,
        help='hidden layer size for Qnet and policy nets'
    )
    parser.add_argument(
        '--hidden-size',
        type=int,
        default=256,
        help='hidden layer size for Qnet and policy nets'
    )
    parser.add_argument(
        '--recurrent-hidden-state',
        type=int,
        default=1024,
        help='recurrent hidden state size'
    )
    parser.add_argument(
        '--max-replay-buffer-size',
        type=int,
        default=1e5,
        help='Replay Buffer Size'
    )
    parser.add_argument(
        '--rnn-seq-len',
        type=int,
        default=20,
        help='RNN Train length'
    )
    parser.add_argument(
        '--policy-lr',
        type=float,
        default=5e-4,
        help='Policy Learning Rate'
    )
    parser.add_argument(
        '--qf-lr',
        type=float,
        default=1e-3,
        help='Q Net Learning Rate'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='batch-size for training'
    )
    parser.add_argument(
        '--max-train-path-len',
        type=int,
        default=500,
        help='max_train_path_len'
    )
    parser.add_argument(
        '--eval-steps',
        type=int,
        default=500,
        help='num eval steps per epoch'
    )
    parser.add_argument(
        '--num-train-epochs',
        type=int,
        default=1e5,
        help='total training epochs'
    )
    parser.add_argument(
        '--num-steps-per-loop',
        type=int,
        default=500,
        help='num expl steps per train loop'
    )
    parser.add_argument(
        '--num-trains-per-loop',
        type=int,
        default=50,
        help='num trains per train loop'
    )
    parser.add_argument(
        '--num-train-loops-per-epoch',
        type=int,
        default=10,
        help='num train loops per epoch'
    )
    parser.add_argument(
        '--befor-traning',
        type=int,
        default=4000,
        help='collect steps before begin training'
    )
    parser.add_argument(
        '--save-path',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)'
    )
    parser.add_argument(
        '--no-cuda',
        action='store_true', default=False,
        help='disables CUDA training'
    )

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args





