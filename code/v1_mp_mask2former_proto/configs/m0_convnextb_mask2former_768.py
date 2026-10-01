# M0: standalone v04 protocol + native Mask2Former 6/9/100 decoder.
# Structure adapted from the shared MMSeg reference; no ADE20K config inheritance.

checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_in21k_20220301-262fd037.pth'

crop_size = (768, 768)

custom_hooks = [{'type': 'AICRunStatsHook'}]

custom_imports = {'allow_failed_imports': False, 'imports': ['mmdet.models', 'mmpretrain.models', 'aicseg']}

data_preprocessor = {'bgr_to_rgb': True,
 'mean': [123.675, 116.28, 103.53],
 'pad_val': 0,
 'seg_pad_val': 255,
 'size': (768, 768),
 'std': [58.395, 57.12, 57.375],
 'type': 'SegDataPreProcessor'}

data_root = '../../data'

default_hooks = {'checkpoint': {'by_epoch': False,
                'interval': 4000,
                'max_keep_ckpts': 2,
                'rule': 'greater',
                'save_best': 'mIoU',
                'type': 'CheckpointHook'},
 'logger': {'interval': 50, 'log_metric_by_epoch': False, 'type': 'LoggerHook'},
 'param_scheduler': {'type': 'ParamSchedulerHook'},
 'sampler_seed': {'type': 'DistSamplerSeedHook'},
 'timer': {'type': 'IterTimerHook'},
 'visualization': {'type': 'SegVisualizationHook'}}

default_scope = 'mmseg'

env_cfg = {'cudnn_benchmark': True,
 'dist_cfg': {'backend': 'nccl'},
 'mp_cfg': {'mp_start_method': 'spawn', 'opencv_num_threads': 0}}

experiment_name = 'v1_mp_m0_reference_768'

load_from = None

log_level = 'INFO'

log_processor = {'by_epoch': False}

max_iters = 40000

model = {'backbone': {'arch': 'base',
              'drop_path_rate': 0.4,
              'gap_before_final_norm': False,
              'init_cfg': {'checkpoint': 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_in21k_20220301-262fd037.pth',
                           'prefix': 'backbone.',
                           'type': 'Pretrained'},
              'layer_scale_init_value': 1.0,
              'out_indices': [0, 1, 2, 3],
              'type': 'mmpretrain.ConvNeXt'},
 'data_preprocessor': {'bgr_to_rgb': True,
                       'mean': [123.675, 116.28, 103.53],
                       'pad_val': 0,
                       'seg_pad_val': 255,
                       'size': (768, 768),
                       'std': [58.395, 57.12, 57.375],
                       'type': 'SegDataPreProcessor'},
 'decode_head': {'type': 'AICMask2FormerHead',
                 'in_channels': [128, 256, 512, 1024],
                 'strides': [4, 8, 16, 32],
                 'feat_channels': 256,
                 'out_channels': 256,
                 'num_classes': 8,
                 'num_queries': 100,
                 'num_transformer_feat_level': 3,
                 'align_corners': False,
                 'pixel_decoder': {'type': 'mmdet.MSDeformAttnPixelDecoder',
                                   'num_outs': 3,
                                   'norm_cfg': {'type': 'GN', 'num_groups': 32},
                                   'act_cfg': {'type': 'ReLU'},
                                   'encoder': {'num_layers': 6,
                                               'layer_cfg': {'self_attn_cfg': {'embed_dims': 256,
                                                                               'num_heads': 8,
                                                                               'num_levels': 3,
                                                                               'num_points': 4,
                                                                               'im2col_step': 64,
                                                                               'dropout': 0.0,
                                                                               'batch_first': True,
                                                                               'norm_cfg': None,
                                                                               'init_cfg': None},
                                                             'ffn_cfg': {'embed_dims': 256,
                                                                         'feedforward_channels': 1024,
                                                                         'num_fcs': 2,
                                                                         'ffn_drop': 0.0,
                                                                         'act_cfg': {'type': 'ReLU',
                                                                                     'inplace': True}}},
                                               'init_cfg': None},
                                   'positional_encoding': {'num_feats': 128, 'normalize': True},
                                   'init_cfg': None},
                 'enforce_decoder_input_project': False,
                 'positional_encoding': {'num_feats': 128, 'normalize': True},
                 'transformer_decoder': {'return_intermediate': True,
                                         'num_layers': 9,
                                         'layer_cfg': {'self_attn_cfg': {'embed_dims': 256,
                                                                         'num_heads': 8,
                                                                         'attn_drop': 0.0,
                                                                         'proj_drop': 0.0,
                                                                         'dropout_layer': None,
                                                                         'batch_first': True},
                                                       'cross_attn_cfg': {'embed_dims': 256,
                                                                          'num_heads': 8,
                                                                          'attn_drop': 0.0,
                                                                          'proj_drop': 0.0,
                                                                          'dropout_layer': None,
                                                                          'batch_first': True},
                                                       'ffn_cfg': {'embed_dims': 256,
                                                                   'feedforward_channels': 2048,
                                                                   'num_fcs': 2,
                                                                   'act_cfg': {'type': 'ReLU',
                                                                               'inplace': True},
                                                                   'ffn_drop': 0.0,
                                                                   'dropout_layer': None,
                                                                   'add_identity': True}},
                                         'init_cfg': None},
                 'loss_cls': {'type': 'mmdet.CrossEntropyLoss',
                              'use_sigmoid': False,
                              'loss_weight': 2.0,
                              'reduction': 'mean',
                              'class_weight': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1]},
                 'loss_mask': {'type': 'mmdet.CrossEntropyLoss',
                               'use_sigmoid': True,
                               'reduction': 'mean',
                               'loss_weight': 5.0},
                 'loss_dice': {'type': 'mmdet.DiceLoss',
                               'use_sigmoid': True,
                               'activate': True,
                               'reduction': 'mean',
                               'naive_dice': True,
                               'eps': 1.0,
                               'loss_weight': 5.0},
                 'train_cfg': {'num_points': 12544,
                               'oversample_ratio': 3.0,
                               'importance_sample_ratio': 0.75,
                               'assigner': {'type': 'mmdet.HungarianAssigner',
                                            'match_costs': [{'type': 'mmdet.ClassificationCost',
                                                             'weight': 2.0},
                                                            {'type': 'mmdet.CrossEntropyLossCost',
                                                             'weight': 5.0,
                                                             'use_sigmoid': True},
                                                            {'type': 'mmdet.DiceCost',
                                                             'weight': 5.0,
                                                             'pred_act': True,
                                                             'eps': 1.0}]},
                               'sampler': {'type': 'mmdet.MaskPseudoSampler'}},
                 'num_things_classes': 0,
                 'num_stuff_classes': 8,
                 'ignore_index': 255},
 'test_cfg': {'crop_size': (768, 768), 'mode': 'slide', 'stride': (512, 512)},
 'train_cfg': {},
 'type': 'EncoderDecoder'}

norm_cfg = {'num_groups': 32, 'requires_grad': True, 'type': 'GN'}

num_classes = 8

optim_wrapper = {'accumulative_counts': 4,
 'constructor': 'AICMask2FormerOptimizerConstructor',
 'loss_scale': 'dynamic',
 'optimizer': {'betas': (0.9, 0.999), 'lr': 0.0001, 'type': 'AdamW', 'weight_decay': 0.05},
 'paramwise_cfg': {'decay_rate': 0.9, 'decay_type': 'stage_wise', 'num_layers': 12},
 'type': 'AmpOptimWrapper'}

param_scheduler = [{'begin': 0, 'by_epoch': False, 'end': 750, 'start_factor': 1e-06, 'type': 'LinearLR'},
 {'begin': 750, 'by_epoch': False, 'end': 40000, 'eta_min': 0.0, 'power': 1.0, 'type': 'PolyLR'}]

randomness = {'deterministic': False, 'seed': 2026}

resume = False

test_cfg = {'type': 'TestLoop'}

test_dataloader = {'batch_size': 1,
 'dataset': {'data_prefix': {'img_path': 'test/images'},
             'data_root': '../../data',
             'pipeline': [{'type': 'LoadImageFromFile'}, {'type': 'PackSegInputs'}],
             'test_mode': True,
             'type': 'AICDataset'},
 'num_workers': 4,
 'persistent_workers': True,
 'sampler': {'shuffle': False, 'type': 'DefaultSampler'}}

test_evaluator = {'format_only': True, 'iou_metrics': ['mIoU'], 'output_dir': 'predictions', 'type': 'AICIoUMetric'}

test_pipeline = [{'type': 'LoadImageFromFile'}, {'type': 'PackSegInputs'}]

train_cfg = {'max_iters': 40000, 'type': 'IterBasedTrainLoop', 'val_interval': 4000}

train_dataloader = {'batch_size': 2,
 'dataset': {'data_prefix': {'img_path': 'train/images', 'seg_map_path': 'train/masks'},
             'data_root': '../../data',
             'pipeline': [{'type': 'LoadImageFromFile'},
                          {'type': 'LoadAnnotations'},
                          {'keep_ratio': True,
                           'ratio_range': (0.5, 2.0),
                           'scale': (1024, 1024),
                           'type': 'RandomResize'},
                          {'cat_max_ratio': 0.75, 'crop_size': (768, 768), 'type': 'RandomCrop'},
                          {'type': 'RandomD4'},
                          {'type': 'PhotoMetricDistortion'},
                          {'type': 'PackSegInputs'}],
             'type': 'AICDataset'},
 'num_workers': 4,
 'persistent_workers': True,
 'sampler': {'shuffle': True, 'type': 'InfiniteSampler'}}

train_pipeline = [{'type': 'LoadImageFromFile'},
 {'type': 'LoadAnnotations'},
 {'keep_ratio': True, 'ratio_range': (0.5, 2.0), 'scale': (1024, 1024), 'type': 'RandomResize'},
 {'cat_max_ratio': 0.75, 'crop_size': (768, 768), 'type': 'RandomCrop'},
 {'type': 'RandomD4'},
 {'type': 'PhotoMetricDistortion'},
 {'type': 'PackSegInputs'}]

train_size = 5596

val_cfg = {'type': 'ValLoop'}

val_dataloader = {'batch_size': 1,
 'dataset': {'data_prefix': {'img_path': 'val/images', 'seg_map_path': 'val/masks'},
             'data_root': '../../data',
             'pipeline': [{'type': 'LoadImageFromFile'},
                          {'type': 'LoadAnnotations'},
                          {'type': 'PackSegInputs'}],
             'test_mode': True,
             'type': 'AICDataset'},
 'num_workers': 4,
 'persistent_workers': True,
 'sampler': {'shuffle': False, 'type': 'DefaultSampler'}}

val_evaluator = {'iou_metrics': ['mIoU'], 'type': 'AICIoUMetric'}

val_interval = 4000

val_pipeline = [{'type': 'LoadImageFromFile'}, {'type': 'LoadAnnotations'}, {'type': 'PackSegInputs'}]

vis_backends = [{'type': 'LocalVisBackend'}]

visualizer = {'name': 'visualizer', 'type': 'SegLocalVisualizer', 'vis_backends': [{'type': 'LocalVisBackend'}]}

work_dir = 'runs/v1_mp_m0_reference_768'
