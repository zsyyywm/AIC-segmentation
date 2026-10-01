"""ConvNeXt-L + UPerNet with D4, Lovasz loss and GroupNorm."""

custom_imports = dict(
    imports=['mmpretrain.models', 'aicseg'], allow_failed_imports=False)
default_scope = 'mmseg'

crop_size = (768, 768)
num_classes = 8
experiment_name = 'v02_convnextl_d4_lovasz_gn_768'
train_size = 5596
data_root = '../../data'
max_iters = 80000
val_interval = 8000


norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='mmpretrain.ConvNeXt',
        arch='large',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
#LayerScale 是 ConvNeXt 借鉴 Transformer 的一个设计：
# 在每个残差分支的输出上，乘一个可学习的缩放系数。
        layer_scale_init_value=1.0,
#GAP = Global Average Pooling（全局平均池化）
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='../convnext-large_3rdparty_in21k_20220301-e6e0ea0a.pth',
            prefix='backbone.')),
    decode_head=dict(
        type='UPerHead',
        in_channels=[192, 384, 768, 1536],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        ignore_index=255,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                avg_non_ignore=True),
            dict(
                type='LovaszLoss',
                loss_type='multi_class',
                classes='present',
                per_image=False,
                reduction='none',
                loss_weight=0.5,
                loss_name='loss_lovasz')
        ]),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        ignore_index=255,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4,
            avg_non_ignore=True)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(512, 512)))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(1024, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomD4'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='AICDataset',
        data_root=data_root,
        data_prefix=dict(
            img_path='train/images', seg_map_path='train/masks'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='AICDataset',
        data_root=data_root,
        data_prefix=dict(img_path='val/images', seg_map_path='val/masks'),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='AICDataset',
        data_root=data_root,
        data_prefix=dict(img_path='test/images'),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AICIoUMetric', iou_metrics=['mIoU'])
test_evaluator = dict(
    type='AICIoUMetric',
    iou_metrics=['mIoU'],
    format_only=True,
    output_dir='predictions')

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    accumulative_counts=8,
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        decay_rate=0.9, decay_type='stage_wise', num_layers=12),
    constructor='LearningRateDecayOptimizerConstructor')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=max_iters,
        eta_min=0.0,
        by_epoch=False),
]

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iters, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=val_interval,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
custom_hooks = [dict(type='AICRunStatsHook')]

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=2026, deterministic=False)
work_dir = 'runs/v02_convnextl_d4_lovasz_gn_768'
