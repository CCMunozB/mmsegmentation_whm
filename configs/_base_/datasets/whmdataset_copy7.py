# dataset settings
dataset_type = 'WMHDataset'
data_root = 'data/WMH_6'
crop_size = (224, 224)
train_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='tifffile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomRotate', prob=0.5, degree=20.0, seg_pad_val=0),
    dict(type='RandomFlip', prob=0.5),
    dict(type='TranslateX', prob=0.5, img_border_value=0),
    dict(type='TranslateY', prob=0.5, img_border_value=0),
    dict(type='ShearX', prob=0.6, max_mag=10.0, img_border_value=0),
    dict(type='ShearY', prob=0.6, max_mag=10.0, img_border_value=0),
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
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='imgs/train',
            seg_map_path='label/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='imgs/val', seg_map_path='label/val'),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='imgs/test', seg_map_path='label/test'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice', 'mIoU'], prefix="dice")
test_evaluator = val_evaluator