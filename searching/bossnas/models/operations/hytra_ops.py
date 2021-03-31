import math

from timm.models.layers import create_attn
from torch import nn


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=128):
        super(Attention, self).__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(x).view(b, 3 * self.heads, self.dim_head, h * w).chunk(3, dim=1)

        q *= self.scale
        attn = q.transpose(-2, -1) @ k
        attn = attn.softmax(dim=-1)

        out = (v @ attn).reshape(b, -1, h, w)
        return out


class PEG(nn.Module):
    """
    modified from PEG in https://github.com/Meituan-AutoML/CPVT
    """
    def __init__(self, dim, stride):
        super(PEG, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim)

    def forward(self, x):
        # x = x + self.conv(x)
        return self.conv(x)


class ResAtt(nn.Module):
    """
    modified from BoTNet block in https://github.com/lucidrains/bottleneck-transformer-pytorch
    which is translated from tensorflow code
    https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
    """
    def __init__(self, dim, fmap_size, dim_out, proj_factor, heads=4, dim_head=128,
                 act_layer=nn.ReLU, up_dim=None):
        super().__init__()
        self.fmap_size = fmap_size
        self.inc = dim
        # shortcut
        if up_dim is not None:
            self.shortcut_d = nn.Sequential(
                nn.Conv2d(up_dim, dim_out, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dim_out),
                act_layer(inplace=True))
        if dim != dim_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(dim, dim_out, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(dim_out),
                act_layer(inplace=True))
        else:
            self.shortcut = nn.Identity()

        # contraction and expansion
        attn_dim_in = dim_out // proj_factor
        attn_dim_out = heads * dim_head
        if up_dim is not None:
            self.proj_d = nn.Sequential(nn.Conv2d(up_dim, attn_dim_in, 1, bias=False),
                                        PEG(attn_dim_in, stride=2),
                                        nn.BatchNorm2d(attn_dim_in))
        self.proj = nn.Sequential(nn.Conv2d(dim, attn_dim_in, 1, bias=False),
                                  PEG(attn_dim_in, stride=1),
                                  nn.BatchNorm2d(attn_dim_in))

        self.net = nn.Sequential(
            act_layer(inplace=True),
            Attention(
                dim=attn_dim_in,
                heads=heads,
                dim_head=dim_head),
            nn.BatchNorm2d(attn_dim_out),
            act_layer(inplace=True),
            nn.Conv2d(attn_dim_out, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out))

        # init last batch norm gamma to zero
        nn.init.zeros_(self.net[-1].weight)

        # final activation
        self.activation = act_layer(inplace=True)

    def zero_init_last_bn(self):
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, x):
        fmap_size = x.shape[-1]
        if fmap_size > self.fmap_size:
            shortcut = self.shortcut_d(x)
            x = self.proj_d(x)
        else:
            shortcut = self.shortcut(x)
            x = self.proj(x)
        x = self.net(x)
        x += shortcut
        return self.activation(x)


class ResConv(nn.Module):
    """
    modified from ResNet block in timm/models/resnet.py
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, cardinality=1, base_width=64, reduce_first=1, dilation=1,
                 first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, attn_layer=None,
                 aa_layer=None, drop_block=None, drop_path=None, fmap_size=56, up_inplanes=None):
        super(ResConv, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)
        self.fmap_size = fmap_size

        if up_inplanes is not None:
            self.downsample_d = nn.Sequential(
                nn.Conv2d(up_inplanes, outplanes, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(outplanes),
                act_layer(inplace=True))
        if inplanes != outplanes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(outplanes),
                act_layer(inplace=True))
        else:
            self.downsample = nn.Identity()

        if up_inplanes is not None:
            self.conv1_d = nn.Conv2d(up_inplanes, first_planes, kernel_size=1, bias=False)
            self.peg_d = PEG(first_planes, stride=2)
            self.bn1_d = norm_layer(first_planes)
        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.peg = PEG(first_planes, stride=1)
        self.bn1 = norm_layer(first_planes)

        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.aa = aa_layer(channels=width, stride=stride) if use_aa else None

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path
        self.inc = inplanes
        self.zero_init_last_bn()

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        fmap_size = x.shape[-1]

        if fmap_size > self.fmap_size:
            residual = self.downsample_d(x)
            x = self.conv1_d(x)
            x = self.peg_d(x)
            x = self.bn1_d(x)
        else:
            residual = self.downsample(x)
            x = self.conv1(x)
            x = self.peg(x)
            x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)
        if self.aa is not None:
            x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x
