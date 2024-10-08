# dataset settings
dataset_type = 'WMHDatasetMulti'
data_root = 'data/WMH_Multi'
crop_size = (224, 224)
train_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='tifffile'),
    dict(type='LoadMultiAnnotations'),
    dict(type='RandomMultiRotFlip', rotate_prob=0.5, flip_prob=0.5, degree=15.0),
    dict(type='MultiShearXY', prob=0.6, max_mag=15.0, img_border_value=0),
    dict(type='MultiTranslateXY', prob=0.6, img_border_value=0),
    dict(type='PackMultiSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='tifffile'),
    dict(type='LoadMultiAnnotations'),
    dict(type='PackMultiSegInputs')
]

tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, imdecode_backend='tifffile'),
    dict(
        type='TestTimeAug',
        transforms=[
             [dict(type='LoadMultiAnnotations')], [dict(type='PackMultiSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=5,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=80000,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(
                img_path='imgs/train',
                seg_map_path='label1/train', # Coordinar Input-Output con labels y prediction features
                seg_map_path2='label2/train'),
            pipeline=train_pipeline)))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='imgs/val', seg_map_path='label1/val', seg_map_path2='label2/val'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice', 'mFscore'], prefix="dice")
test_evaluator = val_evaluator