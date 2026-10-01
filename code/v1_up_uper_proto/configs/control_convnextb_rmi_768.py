checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_in21k_20220301-262fd037.pth'
crop_size = (
    768,
    768,
)
custom_hooks = [
    dict(type='AICRunStatsHook'),
]
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'mmpretrain.models',
        'aicseg',
    ])
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        768,
        768,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = '../../data'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=4000,
        max_keep_ckpts=2,
        rule='greater',
        save_best='mIoU',
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0))
experiment_name = 'v1_convnextb_rmi_control_768'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
max_iters = 40000
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        ignore_index=255,
        in_channels=512,
        in_index=2,
        loss_decode=dict(
            avg_non_ignore=True,
            loss_weight=0.4,
            type='CrossEntropyLoss',
            use_sigmoid=False),
        norm_cfg=dict(num_groups=32, requires_grad=True, type='GN'),
        num_classes=8,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        arch='base',
        drop_path_rate=0.4,
        gap_before_final_norm=False,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_in21k_20220301-262fd037.pth',
            prefix='backbone.',
            type='Pretrained'),
        layer_scale_init_value=1.0,
        out_indices=[
            0,
            1,
            2,
            3,
        ],
        type='mmpretrain.ConvNeXt'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            768,
            768,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=512,
        dropout_ratio=0.1,
        ignore_index=255,
        in_channels=[
            128,
            256,
            512,
            1024,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=[
            dict(
                avg_non_ignore=True,
                loss_weight=1.0,
                type='CrossEntropyLoss',
                use_sigmoid=False),
            dict(
                covariance_epsilon=1e-06,
                loss_name='loss_rmi',
                loss_weight=0.5,
                num_classes=8,
                pool_size=4,
                radius=3,
                type='RegionMutualInformationLoss'),
        ],
        norm_cfg=dict(num_groups=32, requires_grad=True, type='GN'),
        num_classes=8,
        pool_scales=(
            1,
            2,
            3,
            6,
        ),
        type='UPerHead'),
    test_cfg=dict(crop_size=(
        768,
        768,
    ), mode='slide', stride=(
        512,
        512,
    )),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(num_groups=32, requires_grad=True, type='GN')
num_classes = 8
optim_wrapper = dict(
    accumulative_counts=4,
    constructor='LearningRateDecayOptimizerConstructor',
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(decay_rate=0.9, decay_type='stage_wise', num_layers=12),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=750, start_factor=1e-06, type='LinearLR'),
    dict(
        begin=750,
        by_epoch=False,
        end=40000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
randomness = dict(deterministic=False, seed=2026)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='test/images'),
        data_root='../../data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='PackSegInputs'),
        ],
        test_mode=True,
        type='AICDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    format_only=True,
    iou_metrics=[
        'mIoU',
    ],
    output_dir='predictions',
    type='AICIoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=40000, type='IterBasedTrainLoop', val_interval=4000)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_prefix=dict(img_path='train/images', seg_map_path='train/masks'),
        data_root='../../data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    1024,
                    1024,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    768,
                    768,
                ), type='RandomCrop'),
            dict(type='RandomD4'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='AICDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            1024,
            1024,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        768,
        768,
    ), type='RandomCrop'),
    dict(type='RandomD4'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
train_size = 5596
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='val/images', seg_map_path='val/masks'),
        data_root='../../data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        test_mode=True,
        type='AICDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='AICIoUMetric')
val_interval = 4000
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'runs/v1_convnextb_rmi_control_768'
