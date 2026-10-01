_base_ = './m0_convnextb_mask2former_768.py'

experiment_name = 'v1_mp_convnextb_mask2former_proto_768'
work_dir = 'runs/v1_mp_convnextb_mask2former_proto_768'

model = dict(
    decode_head=dict(
        type='AICPrototypeMask2FormerHead', feature_stride=4,
        prototype_cfg=dict(
            feature_dim=256, num_classes=8, prototypes_per_class=4,
            momentum=0.99, temperature=0.1, gate_init=1e-3,
            max_update_pixels_per_class=2048, enabled=True,
            ignore_index=255, eps=1e-6)))

custom_hooks = [dict(type='AICRunStatsHook'),
                dict(type='PrototypeDiagnosticsHook', interval=50,
                     core_path='decode_head.prototype_adapter.core')]
