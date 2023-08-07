# dataset settings
dataset_type = 'WMHDatasetMulti'
data_root = 'data/WMH_Multi'
crop_size = (224, 224)
train_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='tifffile'),
    dict(type='LoadMultiAnnotations'),
    dict(type='RandomRotate', prob=0.5, degree=15.0),
    dict(type='RandomFlip', prob=0.5),
    #dict(type='RandAugment', aug_num=2),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='tifffile'),
    dict(type='LoadMultiAnnotations'),
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
            ], [dict(type='LoadMultiAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=5,
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

val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice', 'mFscore'], prefix="dice")
test_evaluator = val_evaluator