import numpy as np
import torch.nn as nn

from bossnas.models.operations.operation_dict import OPS, reset
from openselfsup.models.registry import BACKBONES

PRIMITIVES = ['MB6_3x3_se0.25',
              'MB6_5x5_se0.25',
              'MB3_3x3_se0.25',
              'MB3_5x5_se0.25',
              ]


def uniform_random_op_encoding(num_of_ops, layers):
    return np.random.randint(0, num_of_ops, layers)


def fair_random_op_encoding(num_of_ops, layers):
    # return alist 
    encodings = np.zeros((layers, num_of_ops), dtype=np.int8)
    for i in range(layers):
        encodings[:][i] = np.random.choice(np.arange(0, num_of_ops),
                                           size=num_of_ops,
                                           replace=False)
    return encodings.T.tolist()


def get_path(str, num):
    if (num == 1):
        for x in str:
            yield x
    else:
        for x in str:
            for y in get_path(str, num - 1):
                yield x + y


def all_op_encoding(num_of_ops, layers):
    # return alist
    encodings = []
    strKey = ""
    for x in range(num_of_ops):
        strKey += str(x)
    for path in get_path(strKey, layers):
        encodings.append([int(op) for op in path])
    return encodings


class MixOps(nn.Module):
    def __init__(self, inc, outc, stride, to_dispatch=False, init_op_index=None, hidden_outc=None, downsample_share=False):
        assert to_dispatch == (init_op_index is not None)
        super(MixOps, self).__init__()
        self._mix_ops = nn.ModuleList()
        if inc != outc or stride == 2:
            downsample_layers = []
            if stride == 2:
                downsample_layers += [nn.AvgPool2d(2, 2, ceil_mode=True, count_include_pad=False)]
            if inc != outc:
                downsample_layers += [nn.Conv2d(inc, outc, kernel_size=1, stride=1, bias=False),
                                      nn.BatchNorm2d(outc)]
            self.downsample = nn.Sequential(*downsample_layers)
        else:
            self.downsample = None
        if downsample_share:
            downsamples = [self.downsample] * len(PRIMITIVES)
        else:
            downsamples = [self.downsample for _ in range(len(PRIMITIVES))]

        if to_dispatch:
            if PRIMITIVES[init_op_index].endswith('_dual'):
                self._mix_ops.append(OPS[PRIMITIVES[init_op_index]](inc, outc, stride, hidden_outc))
            else:
                self._mix_ops.append(OPS[PRIMITIVES[init_op_index]](inc, outc, stride))

        else:
            for i, prim in enumerate(PRIMITIVES):
                if prim.endswith('_dual'):
                    self._mix_ops.append(OPS[prim](inc, outc, stride, hidden_outc))
                else:
                    self._mix_ops.append(OPS[prim](inc, outc, stride, downsamples[i]))

    def forward(self, x, forward_index=0):
        # Single-path
        return self._mix_ops[forward_index](x)


class Block(nn.Module):
    def __init__(self, inc, hidden_outc, outc, stride, layers, to_dispatch=False, init_op_list=None):
        super(Block, self).__init__()
        init_op_list = init_op_list if init_op_list is not None else [None] * layers  # to_dispatch
        self._block_layers = nn.ModuleList()
        # TODO:
        if layers == 1:
            self._block_layers.append(
                MixOps(inc, outc, stride, to_dispatch, init_op_list[0]))
        else:
            for i in range(layers):
                if i == 0:
                    self._block_layers.append(
                        MixOps(inc, hidden_outc, stride, to_dispatch, init_op_list[i],
                               hidden_outc=hidden_outc))
                elif i == layers - 1:
                    self._block_layers.append(
                        MixOps(hidden_outc, outc, 1, to_dispatch, init_op_list[i],
                               hidden_outc=hidden_outc))
                else:
                    self._block_layers.append(
                        MixOps(hidden_outc, hidden_outc, 1, to_dispatch, init_op_list[i],
                               hidden_outc=hidden_outc))

    def forward(self, x, forwad_list=None):
        assert len(forwad_list) == len(self._block_layers)
        for i, layer in enumerate(self._block_layers):
            x = layer(x, forwad_list[i])
        return x

    def reset_params(self):
        self.apply(reset)


@BACKBONES.register_module
class SupernetMBConv(nn.Module):

    def __init__(self, to_dispatch=False, init_op_list=None, block_layers_num=None):
        super(SupernetMBConv, self).__init__()
        #  hidden_outc / outc /stride / layers
        self.block_cfgs = [[24, 24, 2, 2],
                           [40, 40, 2, 4],
                           [80, 80, 2, 4],
                           [112, 112, 1, 4],
                           [192, 192, 2, 4],
                           [320, 320, 1, 1]  # the last stage is fixed by default
                           ]
        if block_layers_num is not None:
            for i in range(len(self.block_cfgs)):
                self.block_cfgs[i][3] = block_layers_num[i]
        # [origin_outc, outc, stride, num_layers]

        # inc, outc = next(channel_cfg), next(channel_cfg)
        self._to_dis = to_dispatch
        self._op_layers_list = [cfg[-1] for cfg in self.block_cfgs]
        self._init_op_list = init_op_list if init_op_list is not None else [None] * sum(
            self._op_layers_list)  # dispatch params
        self._stem = nn.Sequential(OPS['Conv3x3_BN_swish'](3, 32, 2),
                                   OPS['MB1_3x3_se0.25'](32, 16, 1))
        self._make_block(self.block_cfgs)
        self._num_of_ops = len(PRIMITIVES)

    def init_weights(self):
        pass  # FIXME: Using pytorch default weight init

    def _make_block(self, block_cfgs, inc=16):
        self._blocks = nn.ModuleList()
        block_layer_index = 0
        for sing_cfg in block_cfgs:
            hidden_outc, outc, stride, layers = sing_cfg[0], sing_cfg[1], sing_cfg[2], sing_cfg[3]
            self._blocks.append(
                Block(inc, hidden_outc, outc, stride, layers, to_dispatch=self._to_dis,
                      init_op_list=self._init_op_list[
                                   block_layer_index:block_layer_index + layers]))
            inc = outc
            block_layer_index += layers

    def forward(self, x, start_block, forward_op=None, block_op=False):
        outs = []
        if start_block == 0:
            x = self._stem(x)
        for i, block in enumerate(self._blocks):
            if i < start_block:  # if start_block = 1 , forward will skip the block 0 and 1
                continue
            if block_op:
                x = block(x, forward_op)
            else:
                x = block(x, forward_op[sum(self._op_layers_list[:i]):sum(
                    self._op_layers_list[:(i + 1)])])
            break
        self.visualize_feature = x
        outs.append(x)
        return tuple(outs)

    def set_forward_cfg(self, method='fair', ):  # support method: uniform/fair
        # TODO: support fair
        if self._to_dis:  # stand-alone must be zeros
            forward_op = np.zeros(sum(self._op_layers_list), dtype=int)
        elif method == 'uni':
            forward_op = uniform_random_op_encoding(num_of_ops=self._num_of_ops,
                                                          layers=sum(self._op_layers_list))
        elif method == 'fair':
            forward_op = fair_random_op_encoding(num_of_ops=self._num_of_ops,
                                                    layers=sum(self._op_layers_list))
        else:
            raise NotImplementedError
        return forward_op

    def get_all_path(self, start_block):
        return all_op_encoding(num_of_ops=self._num_of_ops,
                                    layers=self._op_layers_list[start_block])

    def reset_params(self):
        self.apply(reset)

    @classmethod
    def dispatch(cls, init_op_list, block_layers_num=None):
        return cls(True, init_op_list, block_layers_num)

    def step_start_trigger(self):
        '''generate fair choices'''
        pass

    def get_layers(self, block):
        '''get num layers of a block'''
        return self.block_cfgs[block][3]

    def get_block(self, block_num):
        '''get block module to train separately'''
        return self._blocks[block_num]


def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters())  # if p.requires_grad)
    return params_num
