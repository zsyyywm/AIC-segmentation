_base_ = './control_convnextb_rmi_768.py'

experiment_name = 'v1_up_convnextb_uper_proto_768'
work_dir = 'runs/v1_up_convnextb_uper_proto_768'

model = dict(
    decode_head=dict(
        type='PrototypeUPerHead',
        feature_stride=4,
        prototype_cfg=dict(
            feature_dim=256, num_classes=8, prototypes_per_class=4,
            momentum=0.99, temperature=0.1, gate_init=1e-3,
            max_update_pixels_per_class=2048, enabled=True, ignore_index=255)))

custom_hooks = [dict(type='AICRunStatsHook'),
                dict(type='PrototypeDiagnosticsHook', interval=50)]
