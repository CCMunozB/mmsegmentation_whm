_base_ = [
    '../_base_/models/upernet_beit.py', '../_base_/datasets/remosdataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

# model = dict(
#     decode_head=dict(remos_weight=[0.2, 0.1, 0.05, 0.65])
#     )

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone


# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader


# bash tools/dist_train.sh configs/remos/uper_remos_wmh2.py 1