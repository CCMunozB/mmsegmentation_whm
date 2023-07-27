# dataset settings
dataset_type = 'WMHDataset'
data_root = 'data/WMH2'
crop_size = (224, 224)
train_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='tifffile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomRotate', prob=0.5, degree=0.4),
    dict(type='RandomFlip', prob=0.5),
    #dict(type='RandomErasing', n_patches=(5,10), ratio=2.0, img_border_value=0),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='tifffile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, imdecode_backend='tifffile'),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=40000,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(
                img_path='imgs/train',
                seg_map_path='label/train'),
            pipeline=train_pipeline)))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='imgs/val', seg_map_path='label/val'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice'], prefix="dice")
test_evaluator = val_evaluator