# dataset settings
dataset_type = 'WHMDataset'
data_root = 'data'
crop_size = (50, 224, 224)
train_pipeline = [
    dict(type='LoadBiomedicalImageFromFile'),
    dict(type='LoadBiomedicalAnnotation', reduce_zero_label=True),
    dict(type='BioMedical3DRandomCrop', crop_size=crop_size, cat_max_ratio=0.95),
    dict(type='BioMedical3DRandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadBiomedicalImageFromFile'),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadBiomedicalAnnotation', reduce_zero_label=True),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadBiomedicalImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='BioMedical3DRandomCrop', crop_size=crop_size, cat_max_ratio=0.95),
                dict(type='BioMedical3DRandomFlip', prob=0.5),
            ], [dict(type='LoadBiomedicalAnnotation')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator