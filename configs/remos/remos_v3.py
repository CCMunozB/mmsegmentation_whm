_base_ = [
    '../_base_/models/uper_remos_wmh2.py', '../_base_/datasets/remosdataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

model = dict(
    decode_head=dict(
        remos_weight=[0.2, 0.1, 0.05, 0.65]),
    )

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.000125, weight_decay=0.1, betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=2000),
    dict(
        type='PolyLR',
        eta_min=1.25e-6,
        power=0.99,
        begin=2000,
        end=40000,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=36)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader


# bash tools/dist_train.sh configs/remos/uper_remos_wmh2.py 1