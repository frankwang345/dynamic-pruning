import torch.nn.functional as F
import torch.nn as nn
import torch
import types
from torch.distributions import RelaxedOneHotCategorical


class TorchGraph(object):
    def __init__(self):
        self._graph = {}
        self.persistence = {}

    def add_tensor_list(self, name, persist=False):
        self._graph[name] = []
        self.persistence[name] = persist

    def append_tensor(self, name, val):
        self._graph[name].append(val)

    def clear_tensor_list(self, name):
        self._graph[name].clear()

    def get_tensor_list(self, name):
        return self._graph[name]

    def clear_all_tensors(self):
        for k in self._graph.keys():
            if not self.persistence[k]:
                self.clear_tensor_list(k)


default_graph = TorchGraph()
default_graph.add_tensor_list('head_params', True)
default_graph.add_tensor_list('gate_params', True)
default_graph.add_tensor_list('sampled_actions')
default_graph.add_tensor_list('selected_channels')
default_graph.add_tensor_list('temperature', True)


class DecisionHead(nn.Module):
    def __init__(self, in_channels, out_channels, action_num, deterministic=False):
        super(DecisionHead, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.action_num = action_num
        self.deterministic = deterministic
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, action_num, bias=False)
        self.relu = nn.ReLU()
        self.channel_gates = nn.Parameter(torch.ones(action_num, out_channels))

    def head_params(self):
        return [self.fc1.weight]

    def gate_params(self):
        return [self.channel_gates]

    def normalize_weights(self):
        self.fc1.weight.data = F.normalize(self.fc1.weight.data, dim=1)

    def forward(self, x):
        out = self.avgpool(self.relu(x))
        out = out.view(x.shape[0], x.shape[1])
        out = self.fc1(out)
        action_probs = F.softmax(out, dim=1)
        if self.deterministic or not self.training:
            sampled_actions = action_probs.max(1)[1]
            selected_channels = self.channel_gates[sampled_actions]
        else:
            temperature = default_graph._graph['temperature']
            m = RelaxedOneHotCategorical(temperature, action_probs)
            actions = m.rsample()
            onehot_actions = torch.zeros(actions.size()).to(x.device)
            sampled_actions = actions.max(1)[1]
            onehot_actions.scatter_(1, sampled_actions.unsqueeze(1), 1)
            substitute_actions = (onehot_actions - actions).detach() + actions
            selected_channels = torch.mm(substitute_actions, self.channel_gates)

        return sampled_actions, selected_channels


def apply_func(model, module_type, func, **kwargs):
    for m in model.modules():
        if m.__class__.__name__ == module_type:
            func(m, **kwargs)


def replace_func(model, module_type, func):
    for m in model.modules():
        if m.__class__.__name__ == module_type:
            m.forward = types.MethodType(func, m)


def collect_params(m):
    for p in m.head_params():
        default_graph.append_tensor('head_params', p)

    for p in m.gate_params():
        default_graph.append_tensor('gate_params', p)


def set_deterministic_value(m, deterministic):
    m.deterministic = deterministic


def normalize_head_weights(m):
    m.normalize_weights()


def init_decision_convbn(m, action_num):
    m.decision_head = DecisionHead(
        m.conv.in_channels, m.conv.out_channels, action_num
    )


def decision_convbn_forward(self, x):
    out = self.conv(x)
    out = self.bn(out)
    if self.conv.in_channels > 3:
        sampled_actions, selected_channels = self.decision_head(x)

        default_graph.append_tensor('sampled_actions', sampled_actions)
        default_graph.append_tensor('selected_channels', selected_channels)

        out = selected_channels.unsqueeze(2).unsqueeze(3) * out
    out = self.relu(out)
    return out



def init_decision_basicblock(m, action_num):
    m.decision_head = DecisionHead(
        m.conv1.in_channels, m.conv1.out_channels, action_num
    )


def decision_basicblock_forward(self, x):
    sampled_actions, selected_channels = self.decision_head(x)

    default_graph.append_tensor('sampled_actions', sampled_actions)
    default_graph.append_tensor('selected_channels', selected_channels)

    out = self.conv1(x)
    out = self.bn1(out)
    out = selected_channels.unsqueeze(2).unsqueeze(3) * out
    out = F.relu(out)

    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)
    out = F.relu(out)
    return out



def init_decision_bottleneck(m, action_num):
    m.decision_head = DecisionHead(
        m.conv1.in_channels, m.conv1.out_channels, action_num
    )


def decision_bottleneck_forward(self, x):
    residual = x

    out = self.bn1(x)
    out = self.relu(out)

    sampled_actions, selected_channels = self.decision_head(out)

    default_graph.append_tensor('sampled_actions', sampled_actions)
    default_graph.append_tensor('selected_channels', selected_channels)

    out = self.conv1(out)

    out = self.bn2(out)
    out = selected_channels.unsqueeze(2).unsqueeze(3) * out
    out = self.relu(out)
    out = self.conv2(out)

    out = self.bn3(out)
    out = self.relu(out)
    out = self.conv3(out)

    if self.downsample is not None:
        residual = self.downsample(x)

    out += residual

    return out
