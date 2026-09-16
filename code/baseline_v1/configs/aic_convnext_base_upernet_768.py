"""ConvNeXt-B + UPerNet development baseline for the AIC competition."""

_base_ = './aic_convnext_large_upernet_768.py'

experiment_name = 'b0_convnextb_upernet_768'

# Batch 2 sees the same number of samples in 40k iterations as the Large
# config does with batch 1 in 80k iterations: about 14.3 equivalent epochs.
max_iters = 40000
val_interval = 4000

checkpoint_file = (
    'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/'
    'convnext-base_3rdparty_in21k_20220301-262fd037.pth')

model = dict(
    backbone=dict(
        arch='base',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=checkpoint_file,
            prefix='backbone.')),
    decode_head=dict(in_channels=[128, 256, 512, 1024]),
    auxiliary_head=dict(in_channels=512))

train_dataloader = dict(batch_size=2)
optim_wrapper = dict(accumulative_counts=4)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=750),
    dict(
        type='PolyLR',
        power=1.0,
        begin=750,
        end=max_iters,
        eta_min=0.0,
        by_epoch=False),
]

train_cfg = dict(max_iters=max_iters, val_interval=val_interval)
default_hooks = dict(checkpoint=dict(interval=val_interval))
work_dir = 'runs/convnext_base_upernet_768'
