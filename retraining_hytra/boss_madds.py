import numpy as np


def madd_net(net='50', image_res=224, use_s1=False, is_bot=False, use_se=False, include_fc=True):
    num_blocks = {
        '50': [3, 4, 6, 3],
        '101': [3, 4, 23, 3],
        '152': [3, 8, 36, 3],
        '128': [3, 4, 23, 12],
        '77': [3, 4, 6, 12],
        '110': [3, 4, 23, 6],
        '59': [3, 4, 6, 6],
        '350': [3, 44, 66, 3],
        'boss0': [1, 1, 5, 9],
        'boss1': [2, 2, 10, 18],
        'boss2': [3, 3, 15, 27]
    }[net]
    resolution = [image_res // 4, image_res // 4, image_res // 8, image_res // 16]
    input_channels = [64, 256, 512, 1024]
    output_channels = [256, 512, 1024, 2048]
    bottleneck_channels = [64, 128, 256, 512]
    strides = [1, 2, 2, 1] if use_s1 else [1, 2, 2, 2]
    is_conv = [True, True, True, False] if is_bot else [True, True, True, True]
    if 'boss' in net:
        is_conv = []
    se_ratio = 0.0625 if use_se else None

    stem_madds = 7 ** 2 * 3 * 64 * (image_res // 2) ** 2
    stem_params = 7 ** 2 * 3 * 64

    dense_madds = 2048 * 1000
    dense_params = 2048 * 1000

    net_madds = stem_madds + dense_madds if include_fc else stem_madds
    net_params = stem_params + dense_params if include_fc else stem_params

    for i in range(4):
        group_madds, group_params = madd_per_group(
            input_channels[i], bottleneck_channels[i], output_channels[i],
            resolution[i], stride=strides[i], is_conv=is_conv[i],
            se_ratio=se_ratio, num_blocks=num_blocks[i])
        net_madds += group_madds
        net_params += group_params

    return net_madds, net_params


def madd_boss_net(net='boss0', image_res=224, use_s1=False, use_se=False, include_fc=True, encoding=None):
    if encoding is None:
        encoding = [[1], [1], [1, 1, 1, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0]]
    scaled_encoding = {
        'boss0': encoding,
        'boss_r50': [[1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1]],
        'boss_vit': [[], [], [0]*16, []],
        'boss_vit_32': [[], [], [], [0]*16],
        'boss_bot': [[1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [0, 0, 0]],
        'boss_random': [[], [], [0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0], [0, 1, 0, 1, 1]],
        'boss_dna': [[1], [1, 1, 1], [1, 0, 1, 1, 1, 1, 0, 1], [0, 1, 1, 1]],
        'boss_unnas': [[], [1], [1, 1], [1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0]]
    }[net]
    print(scaled_encoding)
    resolution = [image_res // 4, image_res // 4, image_res // 8, image_res // 16]
    input_channels = [64, 256, 512, 1024]
    output_channels = [256, 512, 1024, 2048]
    bottleneck_channels = [64, 128, 256, 512]
    strides = [1, 2, 2, 1] if use_s1 else [1, 2, 2, 2]

    se_ratio = 0.0625 if use_se else None

    stem_madds = 7 ** 2 * 3 * 64 * (image_res // 2) ** 2
    stem_params = 7 ** 2 * 3 * 64

    dense_madds = 2048 * 1000
    dense_params = 2048 * 1000

    net_madds = stem_madds + dense_madds if include_fc else stem_madds
    net_params = stem_params + dense_params if include_fc else stem_params

    choice_is_conv = [False, True]
    for stage_i, stage_encod in enumerate(scaled_encoding):
        if len(stage_encod) == 0:
            net_madds += (resolution[stage_i] // strides[stage_i]) ** 2 * 3 ** 2 * bottleneck_channels[stage_i] ** 2
            net_params += 3 ** 2 * bottleneck_channels[stage_i] ** 2
        for blk_i, choice in enumerate(stage_encod):
            if blk_i == 0:  # First block
                blk_madds, blk_params = madd_per_boss_block(
                    input_channels[stage_i], bottleneck_channels[stage_i], output_channels[stage_i],
                    resolution[stage_i],  stride=strides[stage_i], is_conv=choice_is_conv[choice],
                    se_ratio=se_ratio)
            else:
                blk_madds, blk_params = madd_per_boss_block(
                    output_channels[stage_i], bottleneck_channels[stage_i], output_channels[stage_i],
                    resolution[stage_i] // strides[stage_i], stride=1, is_conv=choice_is_conv[choice], se_ratio=se_ratio)
            net_madds += blk_madds
            net_params += blk_params

    return net_madds, net_params


def madd_per_group(
        input_channels,
        bottleneck_channels,
        output_channels,
        resolution,
        stride=2,
        is_conv=True,
        se_ratio=0.0625,
        num_blocks=3):
    madds_first_group, params_first_group = madd_per_bottleneck_block(input_channels, bottleneck_channels,
                                                                      output_channels, resolution, stride=stride,
                                                                      is_conv=is_conv, se_ratio=se_ratio)
    madds_rest, params_rest = madd_per_bottleneck_block(output_channels, bottleneck_channels, output_channels,
                                                        resolution // stride, stride=1, is_conv=is_conv,
                                                        se_ratio=se_ratio)
    madds_rest *= (num_blocks - 1)
    params_rest *= (num_blocks - 1)
    return madds_first_group + madds_rest, params_first_group + params_rest


def madd_per_bottleneck_block(
        input_channels,
        bottleneck_channels,
        output_channels,
        resolution,
        stride=1,
        kernel_size=3,
        is_conv=True,
        se_ratio=0.0625):
    ### Begin Input 1x1 ###
    input_pointwise_madds = input_channels * bottleneck_channels * resolution ** 2
    input_pointwise_params = input_channels * bottleneck_channels
    ### End Input 1x1 ###

    ### Begin Bottleneck ###
    if not is_conv:
        ### Begin All2All Self-Attention w/ Rel-Pos ####
        attention_pointwise_params = 3 * bottleneck_channels ** 2

        attention_pointwise_madds = 3 * bottleneck_channels ** 2 * resolution ** 2
        attention_logit_qk_madds = bottleneck_channels * (resolution) ** 4
        attention_logit_qr_madds = 2 * (2 * resolution - 1) * bottleneck_channels * (
            resolution) ** 2  # no split relative by default
        attention_logit_madds = attention_logit_qk_madds + attention_logit_qr_madds
        v_accumulation_madds = resolution ** 2 * bottleneck_channels * (resolution) ** 2
        spatial_mixing_madds = attention_pointwise_madds + attention_logit_madds + v_accumulation_madds
        conv_params = 0
        spatial_mixing_params = attention_pointwise_params
        ### End All2All Self-Attention w/ Rel-Pos ###
    else:
        ### Begin Conv Flops ###
        spatial_mixing_madds = (resolution // stride) ** 2 * kernel_size ** 2 * bottleneck_channels ** 2
        conv_params = kernel_size ** 2 * bottleneck_channels ** 2
        attention_pointwise_params = 0
        spatial_mixing_params = conv_params
        ### End Conv Flops ###
    ### End Bottleneck ###

    ### Begin Output 1x1 ###
    output_pointwise_madds = bottleneck_channels * output_channels * (resolution // stride) ** 2
    output_pointwise_params = bottleneck_channels * output_channels
    ### End Output 1x1 ###

    ### Projection Shortcut ###
    if input_channels != output_channels:
        projection_madds = input_channels * output_channels * (resolution // stride) ** 2
        projection_params = input_channels * output_channels
    else:
        projection_madds = 0  # (resolution//stride_after)**2 * output_channels DOUBLE CHECK THIS
        projection_params = 0
    ### End Projection Shortcut ###

    ### Squeeze-Excite ###
    if se_ratio:
        se_contraction_madds = output_channels ** 2 * se_ratio
        se_contraction_params = output_channels ** 2 * se_ratio
        se_expansion_madds = output_channels ** 2 * se_ratio
        se_expansion_params = output_channels ** 2 * se_ratio
        se_params = se_expansion_params + se_contraction_params
        se_madds = se_expansion_madds + se_contraction_madds
    else:
        se_contraction_madds = 0
        se_contraction_params = 0
        se_expansion_madds = 0
        se_expansion_params = 0
        se_params = se_expansion_params + se_contraction_params
        se_madds = se_expansion_madds + se_contraction_madds
    ## End Squeeze-Excite ###

    return (input_pointwise_madds + spatial_mixing_madds + output_pointwise_madds + projection_madds + se_madds,
            input_pointwise_params + spatial_mixing_params + output_pointwise_params + projection_params + se_params)


def madd_per_boss_block(
        input_channels,
        bottleneck_channels,
        output_channels,
        resolution,
        stride=1,
        kernel_size=3,
        is_conv=True,
        se_ratio=0.0625):
    ### Begin Input 1x1 ###
    input_pointwise_madds = input_channels * bottleneck_channels * resolution ** 2
    input_pointwise_params = input_channels * bottleneck_channels
    ### End Input 1x1 ###

    ### Begin PEG 3x3 depthsep conv ###
    input_depthsep_madds = (resolution // stride) ** 2 * kernel_size ** 2 * bottleneck_channels
    input_depthsep_params = kernel_size ** 2 * bottleneck_channels
    ### End PEG 3x3 depthsep conv ###

    ### Begin Bottleneck ###
    if not is_conv:  # No rel pos by default in BossNet
        ### Begin All2All Self-Attention  ####
        attention_pointwise_params = 3 * bottleneck_channels ** 2

        attention_pointwise_madds = 3 * bottleneck_channels ** 2 * (resolution // stride) ** 2
        attention_logit_qk_madds = bottleneck_channels * (resolution // stride) ** 4
        # attention_logit_qr_madds = 2 * (2 * (resolution // stride) - 1) * bottleneck_channels * (resolution // stride) ** 2  # no split relative by default
        attention_logit_madds = attention_logit_qk_madds #+ attention_logit_qr_madds
        v_accumulation_madds = (resolution // stride) ** 2 * bottleneck_channels * (resolution // stride) ** 2
        spatial_mixing_madds = attention_pointwise_madds + attention_logit_madds + v_accumulation_madds
        conv_params = 0
        spatial_mixing_params = attention_pointwise_params
        ### End All2All Self-Attention  ###
    else:
        ### Begin Conv Flops ###
        spatial_mixing_madds = (resolution // stride) ** 2 * kernel_size ** 2 * bottleneck_channels ** 2
        conv_params = kernel_size ** 2 * bottleneck_channels ** 2
        attention_pointwise_params = 0
        spatial_mixing_params = conv_params
        ### End Conv Flops ###
    ### End Bottleneck ###

    ### Begin Output 1x1 ###
    output_pointwise_madds = bottleneck_channels * output_channels * (resolution // stride) ** 2
    output_pointwise_params = bottleneck_channels * output_channels
    ### End Output 1x1 ###

    ### Projection Shortcut ###
    if input_channels != output_channels:
        projection_madds = input_channels * output_channels * (resolution // stride) ** 2
        projection_params = input_channels * output_channels
    else:
        projection_madds = 0  # (resolution//stride_after)**2 * output_channels DOUBLE CHECK THIS
        projection_params = 0
    ### End Projection Shortcut ###

    ### Squeeze-Excite ###
    if se_ratio:
        se_contraction_madds = output_channels ** 2 * se_ratio
        se_contraction_params = output_channels ** 2 * se_ratio
        se_expansion_madds = output_channels ** 2 * se_ratio
        se_expansion_params = output_channels ** 2 * se_ratio
        se_params = se_expansion_params + se_contraction_params
        se_madds = se_expansion_madds + se_contraction_madds
    else:
        se_contraction_madds = 0
        se_contraction_params = 0
        se_expansion_madds = 0
        se_expansion_params = 0
        se_params = se_expansion_params + se_contraction_params
        se_madds = se_expansion_madds + se_contraction_madds
    ## End Squeeze-Excite ###

    return (input_pointwise_madds + input_depthsep_madds + spatial_mixing_madds + output_pointwise_madds + projection_madds + se_madds,
            input_pointwise_params + input_depthsep_params + spatial_mixing_params + output_pointwise_params + projection_params + se_params)


base_mrcnn_config = {'net': '50', 'image_res': 1024, 'include_fc': False}
base_imnet_config = {'net': '50', 'image_res': 224}

configs = {
    # MRCNN
    'res50_mrcnn': {'net': '50', 'image_res': 1024, 'include_fc': False},
    'bot50_mrcnn': {'net': '50', 'image_res': 1024, 'include_fc': False, 'is_bot': True},
    'res101_mrcnn': {'net': '101', 'image_res': 1024, 'include_fc': False},
    'bot101_mrcnn': {'net': '101', 'image_res': 1024, 'include_fc': False, 'is_bot': True},
    'res152_mrcnn': {'net': '152', 'image_res': 1024, 'include_fc': False},
    'bot152_mrcnn': {'net': '152', 'image_res': 1024, 'include_fc': False, 'is_bot': True},

    # ImNet
    'res50_imnet': {'net': '50'},
    'res50_imnet_i256': {'net': '50', 'image_res': 256},

    'se50_imnet': {'net': '50', 'use_se': True},

    'bot50_imnet': {'net': '50', 'is_bot': True},
    'bot_s1_50_imnet': {'net': '50', 'is_bot': True, 'use_s1': True},

    'res101_imnet': {'net': '101'},
    'se101_imnet': {'net': '101', 'use_se': True},

    'res152_imnet': {'net': '152'},
    'se152_imnet': {'net': '152', 'use_se': True},

    'bot7': {'net': '128', 'is_bot': True, 'use_s1': True, 'use_se': True, 'image_res': 384},
    'bot7a': {'net': '128', 'is_bot': True, 'use_s1': True, 'use_se': True, 'image_res': 320},

    'bot6': {'net': '77', 'is_bot': True, 'use_s1': True, 'use_se': True, 'image_res': 320},
    'bot5': {'net': '128', 'is_bot': True, 'use_s1': True, 'use_se': True, 'image_res': 256},
    'bot4': {'net': '110', 'is_bot': True, 'use_s1': True, 'use_se': True, 'image_res': 224},
    'bot3': {'net': '59', 'is_bot': True, 'use_s1': True, 'use_se': True, 'image_res': 224},

    'se152_288': {'net': '152', 'use_se': True, 'image_res': 288},
    'se50_160': {'net': '50', 'use_se': True, 'image_res': 160},
    'se350_320': {'net': '350', 'use_se': True, 'image_res': 320},

    'boss_t0_no_se_224': {'net': 'boss0', 'use_se': False, 'image_res': 224},
    'boss_t0_224': {'net': 'boss0', 'use_se': True, 'image_res': 224},
    'boss_t0_288': {'net': 'boss0', 'use_se': True, 'image_res': 288},
    'boss_t1_224': {'net': 'boss0', 'use_se': True, 'image_res': 224, 'use_s1': True},
    'boss_t1_256': {'net': 'boss0', 'use_se': True, 'image_res': 256, 'use_s1': True},

    'boss_r50': {'net': 'boss_r50', 'use_se': True, 'image_res': 224},
    'boss_vit': {'net': 'boss_vit', 'use_se': True, 'image_res': 224},
    'boss_vit_32': {'net': 'boss_vit_32', 'use_se': True, 'image_res': 224},
    'boss_bot': {'net': 'boss_bot', 'use_se': True, 'image_res': 224},
    'boss_random': {'net': 'boss_random', 'use_se': True, 'image_res': 224},
    'boss_dna': {'net': 'boss_dna', 'use_se': True, 'image_res': 224},
    'boss_unnas': {'net': 'boss_unnas', 'use_se': True, 'image_res': 224},
}

models = ['res50_mrcnn', 'bot50_mrcnn', 'res101_mrcnn', 'res152_mrcnn']

for model in models:
    madds, params = madd_net(**configs[model])
    print('madds for {}'.format(model), madds)
    print('params for {}'.format(model), params)

print('############')
print('ImageNet Stuff')
models = ['res50_imnet', 'bot50_imnet', 'bot_s1_50_imnet',
          'res101_imnet', 'res152_imnet', 'se50_imnet', 'se101_imnet',
          'se152_imnet', 'res50_imnet_i256']

for model in models:
    madds, params = madd_net(**configs[model])
    print('madds for {}'.format(model), madds)
    print('params for {}'.format(model), params)

print('############')
print('ImageNet BoT family Stuff')
models = ['bot3', 'bot4', 'bot5', 'bot6', 'bot7', 'bot7a']

for model in models:
    madds, params = madd_net(**configs[model])
    print('madds for {}'.format(model), madds)
    print('params for {}'.format(model), params)

print('############')
print('ImageNet SE family Stuff')
models = ['se50_160', 'se152_288', 'se350_320']

for model in models:
    madds, params = madd_net(**configs[model])
    print('madds for {}'.format(model), madds)
    print('params for {}'.format(model), params)

print('############')
print('ImageNet BossNet-T family')
models = ['boss_t0_no_se_224', 'boss_r50', 'boss_vit', 'boss_vit_32', 'boss_bot', 'boss_random', 'boss_dna', 'boss_unnas', 'boss_t0_224', 'boss_t0_288', 'boss_t1_224', 'boss_t1_256']

for model in models:
    madds, params = madd_boss_net(**configs[model])
    print('madds for {}'.format(model), madds)
    print('params for {}'.format(model), params)
