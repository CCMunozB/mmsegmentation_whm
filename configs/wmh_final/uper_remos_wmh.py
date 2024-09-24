norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth'
data_preprocessor = dict(
    type='SegDataPreProcessor', size=(
        224,
        224,
    ), pad_val=0, seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        size=(
            224,
            224,
        ),
        pad_val=0,
        seg_pad_val=255),
    pretrained=None,
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth'
        ),
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=192,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[
            2,
            2,
            18,
            2,
        ],
        num_heads=[
            6,
            12,
            24,
            48,
        ],
        strides=(
            4,
            2,
            2,
            2,
        ),
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True)),
    decode_head=dict(
        type='UPerRemosHead',
        in_channels=[
            192,
            384,
            768,
            1536,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        pool_scales=(
            1,
            2,
            3,
            6,
        ),
        channels=512,
        dropout_ratio=0.3,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='LogCoshDiceLoss',
                loss_name='loss_dice',
                loss_weight=1.0,
                class_weight=[
                    0.1,
                    1.1,
                ]),
        ],
        remos_weight=[
            0.25,
            0.125,
            0.0625,
            0.5625,
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'WMHDataset'
data_root = 'data/WMH'
crop_size = (
    224,
    224,
)
train_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='tifffile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomRotate', prob=0.5, degree=15.0, seg_pad_val=0),
    dict(type='RandomFlip', prob=0.5),
    dict(type='ShearX', prob=0.5, max_mag=10.0, img_border_value=0),
    dict(type='ShearY', prob=0.5, max_mag=10.0, img_border_value=0),
    dict(type='TranslateX', prob=0.5, img_border_value=0),
    dict(type='TranslateY', prob=0.5, img_border_value=0),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='tifffile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
tta_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=None,
        imdecode_backend='tifffile'),
    dict(
        type='TestTimeAug',
        transforms=[
            [],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ]),
]
train_dataloader = dict(
    batch_size=36,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=20000,
        dataset=dict(
            type='WMHDataset',
            data_root='data/WMH',
            data_prefix=dict(
                img_path='imgs/train', seg_map_path='label/train'),
            pipeline=[
                dict(type='LoadImageFromFile', imdecode_backend='tifffile'),
                dict(type='LoadAnnotations'),
                dict(
                    type='RandomRotate', prob=0.5, degree=15.0, seg_pad_val=0),
                dict(type='RandomFlip', prob=0.5),
                dict(
                    type='ShearX', prob=0.5, max_mag=10.0, img_border_value=0),
                dict(
                    type='ShearY', prob=0.5, max_mag=10.0, img_border_value=0),
                dict(type='TranslateX', prob=0.5, img_border_value=0),
                dict(type='TranslateY', prob=0.5, img_border_value=0),
                dict(type='PackSegInputs'),
            ])))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='WMHDataset',
        data_root='data/WMH',
        data_prefix=dict(img_path='imgs/val', seg_map_path='label/val'),
        pipeline=[
            dict(type='LoadImageFromFile', imdecode_backend='tifffile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='WMHDataset',
        data_root='data/WMH',
        data_prefix=dict(img_path='imgs/test', seg_map_path='label/test'),
        pipeline=[
            dict(type='LoadImageFromFile', imdecode_backend='tifffile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ]))
val_evaluator = dict(
    type='IoUMetric', iou_metrics=[
        'mDice',
        'mIoU',
    ], prefix='dice')
test_evaluator = dict(
    type='IoUMetric', iou_metrics=[
        'mDice',
        'mIoU',
    ], prefix='dice')
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ],
    name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
tta_model = dict(type='SegTTAModel')
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.001, weight_decay=0.01, betas=(
            0.9,
            0.999,
        )),
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1, decay_mult=1.0),
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-06, by_epoch=False, begin=0,
        end=6000),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=6000,
        end=80000,
        by_epoch=False),
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=200, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
launcher = 'pytorch'
work_dir = './work_dirs/uper_remos_wmh'
