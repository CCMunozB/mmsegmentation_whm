_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/test.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (224, 224)
data_preprocessor = dict(size=crop_size)
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        in_channels=3,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        drop_rate=0.,
        attn_drop_rate=0.,
        patch_norm=True),
    decode_head=dict(
        in_channels=[192, 384, 768, 1536], 
        num_classes=2,
        #out_channels=1,
        dropout_ratio=0.3,
        loss_decode=[
        dict(type='LogCoshDiceLoss', loss_name='loss_cosh', loss_weight=0.8, class_weight=[0.0001, 1.0])
        ]),
    auxiliary_head=dict(
        in_channels=768, 
        num_classes=2,
        #out_channels=1,
        dropout_ratio=0.3,
        num_convs=3,
        loss_decode=[
        dict(type='LogCoshDiceLoss', loss_name='loss_cosh', loss_weight=0.2, class_weight=[0.0001, 1.0]),
        ])
    )

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0006, weight_decay=0.002, betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=2000),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=2000,
        end=80000,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=8)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
