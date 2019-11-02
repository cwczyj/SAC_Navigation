import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import math
from models.distributions import TanhNormal
import os

class MultitaskCNN(nn.Module):
    def __init__(
            self,
            num_classes=191,
            pretrained=True,
            checkpoint_path='models/03_13_h3d_hybrid_cnn.pt'
    ):
        super(MultitaskCNN, self).__init__()

        self.num_classes = num_classes
        self.conv_block1_depth = nn.Sequential(
            nn.Conv2d(3, 8, 5),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))

        self.conv_block1_rgb = nn.Sequential(
            nn.Conv2d(3, 8, 5),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))

        self.conv_block2_depth = nn.Sequential(
            nn.Conv2d(8, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))

        self.conv_block2_rgb = nn.Sequential(
            nn.Conv2d(8, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(16, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(32, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential(
            nn.Conv2d(32, 512, 5),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d())

        self.encoder_seg = nn.Conv2d(512, self.num_classes, 1)
        self.encoder_depth = nn.Conv2d(512, 1, 1)
        self.encoder_ae = nn.Conv2d(512, 3, 1)

        self.score_pool2_seg = nn.Conv2d(16, self.num_classes, 1)
        self.score_pool3_seg = nn.Conv2d(32, self.num_classes, 1)

        self.score_pool2_depth = nn.Conv2d(16, 1, 1)
        self.score_pool3_depth = nn.Conv2d(32, 1, 1)

        self.score_pool2_ae = nn.Conv2d(16, 3, 1)
        self.score_pool3_ae = nn.Conv2d(32, 3, 1)

        self.pretrained = pretrained
        if self.pretrained == True and checkpoint_path is not None and os.path.isfile(checkpoint_path):
            print('Loading CNN weights from %s' % checkpoint_path)
            checkpoint = torch.load(
                checkpoint_path, map_location={'cuda:0': 'cpu'})
            self.load_state_dict(checkpoint['state_dict'])
            for param in self.parameters():
                param.requires_grad = False
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * (
                            m.out_channels + m.in_channels)
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, inputs):
        rgb = inputs[:, :3]
        depth = inputs[:, 3:]

        conv1_rgb = self.conv_block1_rgb(rgb)
        conv1_depth = self.conv_block1_depth(depth)

        conv2_rgb = self.conv_block2_rgb(conv1_rgb)
        conv2_depth = self.conv_block2_depth(conv1_depth)

        conv2_concate = torch.cat((conv2_rgb, conv2_depth), 1)

        conv2 = self.conv_block2(conv2_concate)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)

        return conv4.view(-1, 32 * 10 * 10)


class ImageFeatureNet(nn.Module):
    def __init__(self, checkpoint_path):
        super(ImageFeatureNet, self).__init__()

        cnn_kwargs = {'num_classes': 94, 'pretrained': True, 'checkpoint_path': checkpoint_path}
        self.cnn = MultitaskCNN(**cnn_kwargs)

    def forward(self, input):
        obs = self.cnn(input)

        return obs.detach()


class RNNBase(nn.Module):
    def __init__(self, recurrent_input_size, hidden_size):
        super(RNNBase, self).__init__()

        self._hidden_size = hidden_size
        self._input_size = recurrent_input_size

        self.gru = nn.GRU(self._input_size, self._hidden_size)
        # for name, param in self.gru.named_parameters():
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0.0)
        #     elif 'weight' in name:
        #         nn.init.orthogonal_(param)

    @property
    def hidden_state_size(self):
        return self._hidden_size

    @property
    def output_size(self):
        return self._input_size

    def forward(self, x, hxs):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), hxs.unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a T x N x -1 tensor that has been flatten to T*N x 01
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.contiguous().view(T, N, x.size(1))

            # RNN steps
            hxs = hxs.unsqueeze(0)
            output, hxs = self.gru(x, hxs)

            # flatten
            x = output.contiguous().view(T*N, -1)
            hxs = hxs.unsqueeze(0)

        return x, hxs


class Mlp(nn.Module):
    def __init__(self,
                 hidden_sizes,
                 output_size,
                 input_size,
                 need_rnn=True,
                 recurrent_hidden_size=256
                 ):
        super(Mlp, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.recurrent_hidden_size=recurrent_hidden_size
        self.need_rnn = need_rnn
        self.fcs = []
        self.ly_norms = []
        self.hidden_activations = []

        in_size = self.input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, int(next_size))
            in_size = int(next_size)
            self.__setattr__("fc{}".format(i), fc)

            # for name, param in fc.named_parameters():
            #     if 'bias' in name:
            #         nn.init.constant_(param, 0.1)
            #     elif 'weight' in name:
            #         nn.init.orthogonal_(param)

            self.fcs.append(fc)
            ln = nn.LayerNorm(int(next_size))
            self.__setattr__("layer_norm{}".format(i), ln)
            self.ly_norms.append(ln)
            activation = nn.ELU()
            self.__setattr__("layer_activation{}".format(i), activation)
            self.hidden_activations.append(activation)

        if self.need_rnn:
            self.gru = RNNBase(in_size, self.recurrent_hidden_size)

        self.last_fc = nn.Linear(self.recurrent_hidden_size, output_size)
        self.last_fc.weight.data.uniform_(-3e-3, 3e-3)
        self.last_fc.bias.data.uniform_(-3e-3, 3e-3)
        #self.last_fc_activation = nn.ELU()
        # for name, param in self.last_fc.named_parameters():
        #    if 'bias' in name:
        #        nn.init.constant_(param, 0.1)
        #     elif 'weight' in name:
        #        nn.init.orthogonal_(param)

    def forward(self, input, hidden=None):
        if self.need_rnn:
            assert hidden is not None
        else:
            Bs = input.size(0)
            hidden = torch.zeros([Bs, self.recurrent_hidden_size], device=input.device)

        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if i < len(self.fcs) - 1:
                h = self.ly_norms[i](h)
            h = self.hidden_activations[i](h)

        # through RNN network
        h, hxs = self.gru(h, hidden)

        # output
        output = self.last_fc(h)
        #output = self.last_fc_activation(output)

        return output, hxs


class QNet(nn.Module):
    def __init__(self,
                 action_size,
                 feature_size,
                 output_size,
                 hidden_sizes,
                 recurrent_hidden_size,
                 ):
        super(QNet, self).__init__()

        self.action_size = action_size
        self.feature_size = feature_size
        self.recurrent_hidden_size = recurrent_hidden_size

        self.mlp_rnn_net = Mlp(hidden_sizes, output_size, self.action_size + int(self.feature_size / 2),
                               True, recurrent_hidden_size)

        self.feature_net_fc_layer = nn.Sequential(
            nn.Linear(32 * 10 * 10, self.feature_size), nn.ELU(),
            nn.Linear(self.feature_size, int(self.feature_size / 2)), nn.ELU(),
            nn.Dropout(p=0.5)
        )

    def forward(self, obs_feature, action, hidden):
        feature = self.feature_net_fc_layer(obs_feature)

        concate_feature = torch.cat((feature, action), dim=1)
        output, hxs = self.mlp_rnn_net(concate_feature, hidden)

        return output, hxs


LOG_SIG_MAX = 3
LOG_SIG_MIN = -0.5


class TanhGaussianPolicy(Mlp):
    def __init__(self,
                 hidden_sizes,
                 action_size,
                 obs_feature_size,
                 recurrent_hidden_size
                 ):
        super(TanhGaussianPolicy, self).__init__(hidden_sizes,
                                                 action_size,
                                                 int(obs_feature_size/2),
                                                 True,
                                                 recurrent_hidden_size)

        self.log_std = None
        self.std = None

        last_hidden_size = obs_feature_size
        if len(hidden_sizes) > 0:
            last_hidden_size = hidden_sizes[-1]
        self.last_fc_log_std = nn.Linear(recurrent_hidden_size, action_size)
        self.last_fc_log_std.weight.data.uniform_(-3e-3, 3e-3)
        self.last_fc_log_std.bias.data.uniform_(-3e-3, 3e-3)
        #for name, param in self.last_fc_log_std.named_parameters():
        #    if 'bias' in name:
        #        nn.init.constant_(param, 0.1)
        #    elif 'weight' in name:
        #        nn.init.orthogonal_(param)

        self.feature_net_fc_layer = nn.Sequential(
            nn.Linear(32 * 10 * 10, obs_feature_size), nn.ELU(),
            nn.Linear(obs_feature_size, int(obs_feature_size / 2)), nn.ELU(),
            nn.Dropout(p=0.5)
        )

    def forward(self, obs, hidden=None,
                reparameterize=True, deterministic=False, return_log_prob=False):
        if self.need_rnn:
            assert hidden is not None
        else:
            Bs = input.size(0)
            hidden = torch.zeros([Bs, self.recurrent_hidden_size], device=input.device)

        h = self.feature_net_fc_layer(obs)
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if i < len(self.fcs) - 1:
                h = self.ly_norms[i](h)
            h = self.hidden_activations[i](h)

        # go through RNN
        h, hxs = self.gru(h, hidden)

        mean = self.last_fc(h)

        log_std = self.last_fc_log_std(h)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )

                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, hxs, entropy, std,
            mean_action_log_prob, pre_tanh_value
        )


class EvalPolicy(nn.Module):
    def __init__(self, stochastic_policy):
        super(EvalPolicy, self).__init__()
        self.stochastic_policy = stochastic_policy
        self.recurrent_hidden_size = stochastic_policy.recurrent_hidden_size

    def forward(self, obs, hidden):
        output = self.stochastic_policy(obs, hidden, deterministic=True)
        action = output[0]
        hidden_state = output[4]

        return action, hidden_state
