_base_ = [
    '../_base_/models/uper_remos_wmh.py', '../_base_/datasets/whmdataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

model = dict(
    decode_head=dict(
        remos_weight=[1.5, 1.0, 0.5, 0.7])
    )
# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0025, weight_decay=0.01, betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=.1, 
                             decay_mult=1.0),
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-5, by_epoch=False, begin=0, end=2000),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0,
        begin=2000,
        end=40000,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=36)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
