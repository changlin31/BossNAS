import copy
# base.py
train_cfg = {}
test_cfg = {}
optimizer_config = dict()  # grad_clip, coalesce, bucket_size_mb
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
dist_params = dict(backend='nccl')
cudnn_benchmark = True
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
# model settings
model = dict(
    type='SiameseSupernetsNATS',
    pretrained=None,
    base_momentum=0.99,
    pre_conv=True,
    backbone=dict(
        type='SupernetNATS',
        target='cifar10',     # cifar10/cifar100 on nats
    ),
    start_block=0,
    num_block=3,
    neck=dict(
        type='NonLinearNeckSimCLR',
        in_channels=2048,
        hid_channels=4096,
        out_channels=256,
        num_layers=2,
        sync_bn=False,
        with_bias=True,
        with_last_bn=False,
        with_avg_pool=True),
    head=dict(type='LatentPredictHead',
              size_average=True,
              predictor=dict(type='NonLinearNeckSimCLR',
                             in_channels=256, hid_channels=4096,
                             out_channels=256, num_layers=2, sync_bn=False,
                             with_bias=True, with_last_bn=False, with_avg_pool=False)))
# dataset settings
data_source_cfg = dict(type='NATSCifar10', root='../data/cifar/', return_label=False)
train_dataset_type = 'BYOLDataset'
test_dataset_type = 'StoragedBYOLDataset'
img_norm_cfg = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomHorizontalFlip'),
]
# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
train_pipeline1 = copy.deepcopy(train_pipeline)
train_pipeline2 = copy.deepcopy(train_pipeline)

test_pipeline1 = copy.deepcopy(train_pipeline1)
test_pipeline2 = copy.deepcopy(train_pipeline2)
data = dict(
    imgs_per_gpu=256,     # total 256*4(gpu)*4(interval)=4096
    workers_per_gpu=2,
    train=dict(
        type=train_dataset_type,
        data_source=dict(split='train', **data_source_cfg),
        pipeline1=train_pipeline1, pipeline2=train_pipeline2),
    val=dict(
        type=test_dataset_type,
        data_source=dict(split='test', **data_source_cfg),
        pipeline1=test_pipeline1, pipeline2=test_pipeline2,),
    test=dict(
        type=test_dataset_type,
        data_source=dict(split='test', **data_source_cfg),
        pipeline1=test_pipeline1, pipeline2=test_pipeline2,))

# optimizer
optimizer = dict(type='LARS', lr=0.48, weight_decay=0.000001, momentum=0.9,
                 paramwise_options={
                    '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0., lars_exclude=True),
                    'bias': dict(weight_decay=0., lars_exclude=True)})
# apex
use_fp16 = True
# interval for accumulate gradient
update_interval = 4
optimizer_config = dict(update_interval=update_interval, use_fp16=use_fp16)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=2,
    warmup_ratio=0.0001, # cannot be 0
    warmup_by_epoch=True)
checkpoint_config = dict(interval=1)
# runtime settings
total_epochs = 150
# additional hooks
custom_hooks = [
    dict(type='BYOLHook', end_momentum=1., update_interval=update_interval),
    dict(type='FairPathHook'),
    dict(
        type='ValNATSPathHook',
        dataset=data['val'],
        bn_dataset=data['train'],
        initial=True,
        interval=10,
        optimizer_cfg=optimizer,
        lr_cfg=lr_config,
        imgs_per_gpu=256,
        workers_per_gpu=4,
        epoch_per_stage=50,
        resume_best_path='')  # e.g. 'path_rank/bestpath_2.yml'
]
# resume_from = 'checkpoints/stage3_epoch3.pth'
resume_optimizer = False