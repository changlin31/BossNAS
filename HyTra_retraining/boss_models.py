import torch.nn as nn
from timm.models.layers import create_classifier, DropPath
from timm.models.registry import register_model
from timm.models.resnet import drop_blocks
from timm.models.vision_transformer import _cfg

from boss_candidates.bot_op import ResAtt
from boss_candidates.resnet_op import ResConv


def make_boss_blocks(encoding, channels, inplanes,
                     output_stride=32, reduce_first=1, avg_down=False,
                     down_kernel_size=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                     drop_block_rate=0., drop_path_rate=0., attn_layer=None, last_stride=2, **kwargs):
    stages = []
    feature_info = []
    net_num_blocks = 16
    net_block_idx = 0
    net_stride = 4
    expansion = 4
    dilation = prev_dilation = 1
    heads = [1, 2, 4, 8]

    for stage_idx, (planes, block_encoding, db) in enumerate(zip(channels, encoding, drop_blocks(drop_block_rate))):
        num_blocks = len(block_encoding)
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        stride = last_stride if stage_idx == 3 else stride
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        blocks = []
        if num_blocks == 0:
            blocks.append(nn.Sequential(
                nn.Conv2d(inplanes, planes * expansion, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(planes * expansion),
                act_layer(inplace=True)))
            inplanes = planes * expansion
        else:
            for block_idx in range(num_blocks):
                # downsample = downsample if block_idx == 0 else None
                stride = stride if block_idx == 0 else 1
                block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule

                if block_encoding[block_idx] == 0:
                    blocks.append(ResAtt(
                        dim=inplanes, dim_out=planes * expansion, attn_dim_in=heads[stage_idx] * 64, stride=stride,
                        heads=heads[stage_idx], dim_head=64, avg_down=avg_down, act_layer=act_layer))
                else:
                    blocks.append(ResConv(
                        inplanes, planes, stride, first_dilation=prev_dilation,
                        drop_path=DropPath(block_dpr) if block_dpr > 0. else None, avg_down=avg_down,
                        act_layer=act_layer, attn_layer=attn_layer))
                prev_dilation = dilation
                inplanes = planes * expansion
                net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info


class BossNet(nn.Module):
    """
    Modified from ResNet class in timm.
    """

    def __init__(self, encoding=None, num_classes=1000, in_chans=3,
                 cardinality=1, base_width=64, stem_width=64, stem_type='',
                 output_stride=32, block_reduce_first=1, down_kernel_size=1, avg_down=False,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None, attn_layer=None, drop_rate=0.0,
                 drop_path_rate=0.,
                 drop_block_rate=0., global_pool='avg', zero_init_last_bn=True, block_args=None, last_stride=2):
        if encoding is None:
            encoding = [[1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1]]
        expansion = 4
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        super(BossNet, self).__init__()

        # Stem
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs_1 = stem_chs_2 = stem_width
            if 'tiered' in stem_type:
                stem_chs_1 = 3 * (stem_width // 4)
                stem_chs_2 = stem_width if 'narrow' in stem_type else 6 * (stem_width // 4)
            self.conv1 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_chs_1, 3, stride=2, padding=1, bias=False),
                norm_layer(stem_chs_1),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs_1, stem_chs_2, 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs_2),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs_2, inplanes, 3, stride=1, padding=1, bias=False)])
        else:
            self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

        # Stem Pooling
        if aa_layer is not None:
            self.maxpool = nn.Sequential(*[
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                aa_layer(channels=inplanes, stride=2)])
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        channels = [64, 128, 256, 512]

        stage_modules, stage_feature_info = make_boss_blocks(
            encoding, channels, inplanes, cardinality=cardinality, base_width=base_width,
            output_stride=output_stride, reduce_first=block_reduce_first, avg_down=avg_down,
            down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer,
            drop_block_rate=drop_block_rate, drop_path_rate=drop_path_rate, attn_layer=attn_layer,
            last_stride=last_stride, **block_args)
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = 512 * expansion
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes,
                                                      pool_type=global_pool)

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, 'zero_init_last_bn'):
                    m.zero_init_last_bn()

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes,
                                                      pool_type=global_pool)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x


@register_model
def bossnet_T0(pretrained=False, **kwargs):
    """Constructs a ResNet-26-D model.
    stem_width=32, stem_type='deep'
    """
    model = BossNet(encoding=[[1], [1], [1, 1, 1, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0]], avg_down=True, act_layer=nn.SiLU,
                    attn_layer='se')
    model.default_cfg = _cfg()
    return model


@register_model
def bossnet_T0_nose(pretrained=False, **kwargs):
    """Constructs a ResNet-26-D model.
    stem_width=32, stem_type='deep'
    """
    model = BossNet(encoding=[[1], [1], [1, 1, 1, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0]], avg_down=True, act_layer=nn.ReLU,
                    attn_layer=None)
    model.default_cfg = _cfg()
    return model


@register_model
def bossnet_T1(pretrained=False, **kwargs):
    """Constructs a ResNet-26-D model.
    stem_width=32, stem_type='deep'
    """
    model = BossNet(encoding=[[1], [1], [1, 1, 1, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0]], avg_down=True, act_layer=nn.SiLU,
                    attn_layer='se', last_stride=1)
    model.default_cfg = _cfg()
    return model
