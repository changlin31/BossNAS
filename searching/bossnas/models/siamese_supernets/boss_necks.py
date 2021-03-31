import torch
import torch.nn as nn
from packaging import version
from mmcv.cnn import kaiming_init, normal_init

from openselfsup.models.registry import NECKS
from openselfsup.models.utils import build_norm_layer


def _init_weights(module, init_linear='normal', std=0.01, bias=0.):
    assert init_linear in ['normal', 'kaiming'], \
        "Undefined init_linear: {}".format(init_linear)
    for m in module.modules():
        if isinstance(m, nn.Linear):
            if init_linear == 'normal':
                normal_init(m, std=std, bias=bias)
            else:
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                            nn.GroupNorm, nn.SyncBatchNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


@NECKS.register_module
class NonLinearNeckSimCLRProject(nn.Module):
    """SimCLR non-linear neck with input projection.

    Structure: fc(no_bias)-bn(has_bias)-[relu-fc(no_bias)-bn(no_bias)].
        The substructures in [] can be repeated. For the SimCLR default setting,
        the repeat time is 1.
    However, PyTorch does not support to specify (weight=True, bias=False).
        It only support \"affine\" including the weight and bias. Hence, the
        second BatchNorm has bias in this implementation. This is different from
        the official implementation of SimCLR.
    Since SyncBatchNorm in pytorch<1.4.0 does not support 2D input, the input is
        expanded to 4D with shape: (N,C,1,1). Not sure if this workaround
        has no bugs. See the pull request here:
        https://github.com/pytorch/pytorch/pull/29626.

    Args:
        num_layers (int): Number of fc layers, it is 2 in the SimCLR default setting.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_layers=2,
                 sync_bn=True,
                 with_bias=False,
                 with_last_bn=True,
                 with_avg_pool=True):
        super(NonLinearNeckSimCLRProject, self).__init__()
        self.sync_bn = sync_bn
        self.with_last_bn = with_last_bn
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.fix_channels = nn.ModuleDict()
        inc_list = [2048, 1024, 512, 256]
        depth_list = range(4)
        for idx, inc in enumerate(inc_list):
            if inc != in_channels:
                in_depth = depth_list[idx]
                out_depth = 0
                stride = 2 ** (in_depth - out_depth)
                fix_channel = nn.Sequential(
                    # nn.Conv2d(inc, in_channels, kernel_size=stride, stride=stride),
                    nn.Conv2d(inc, inc, kernel_size=stride, stride=stride, groups=inc, bias=False),
                    nn.Conv2d(inc, in_channels, kernel_size=1, bias=False),
                    # nn.BatchNorm2d(inc),
                    # nn.ReLU(inplace=True),
                    # nn.Conv2d(inc, in_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(in_channels))
                self.fix_channels[str(inc)] = fix_channel
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if version.parse(torch.__version__) < version.parse("1.4.0"):
            self.expand_for_syncbn = True
        else:
            self.expand_for_syncbn = False

        self.relu = nn.ReLU(inplace=True)
        self.fc0 = nn.Linear(in_channels, hid_channels, bias=with_bias)
        if sync_bn:
            _, self.bn0 = build_norm_layer(
                dict(type='SyncBN'), hid_channels)
        else:
            self.bn0 = nn.BatchNorm1d(hid_channels)

        self.fc_names = []
        self.bn_names = []
        for i in range(1, num_layers):
            this_channels = out_channels if i == num_layers - 1 \
                else hid_channels
            self.add_module(
                "fc{}".format(i),
                nn.Linear(hid_channels, this_channels, bias=with_bias))
            self.fc_names.append("fc{}".format(i))
            if i != num_layers - 1 or self.with_last_bn:
                if sync_bn:
                    self.add_module(
                        "bn{}".format(i),
                        build_norm_layer(dict(type='SyncBN'), this_channels)[1])
                else:
                    self.add_module(
                        "bn{}".format(i),
                        nn.BatchNorm1d(this_channels))
                self.bn_names.append("bn{}".format(i))
            else:
                self.bn_names.append(None)

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def _forward_syncbn(self, module, x):
        assert x.dim() == 2
        if self.expand_for_syncbn:
            x = module(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        else:
            x = module(x)
        return x

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        #  不同路径进入neck的channel调成一样
        inc = x.shape[1]

        if inc != self.in_channels:
            x = self.fix_channels[str(inc)](x)

        if self.with_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        if self.sync_bn:
            x = self._forward_syncbn(self.bn0, x)
        else:
            x = self.bn0(x)
        for fc_name, bn_name in zip(self.fc_names, self.bn_names):
            fc = getattr(self, fc_name)
            x = self.relu(x)
            x = fc(x)
            if bn_name is not None:
                bn = getattr(self, bn_name)
                if self.sync_bn:
                    x = self._forward_syncbn(bn, x)
                else:
                    x = bn(x)
        return [x]

