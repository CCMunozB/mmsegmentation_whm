norm_cfg = dict(type='SyncBN', requires_grad=True)


checkpoint_file = 'https://github.com/addf400/files/releases/download/v1.0/beit_large_patch16_224_pt22k_ft22k.pth'
data_preprocessor = dict(
    type='SegDataPreProcessor',
    #mean=[123.675, 116.28, 103.53],
    #std=[58.395, 57.12, 57.375],
    #bgr_to_rgb=True,
    size=(224, 224),
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        type='BEiT',
        img_size=(224, 224),
        patch_size=16,
        in_channels=3,
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        mlp_ratio=4,
        out_indices=[7, 11, 15, 23],
        qv_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        init_values=1e-6),
    neck=dict(type='Feature2Pyramid', embed_dim=1024, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type='UPerRemosHead',
        in_channels=[1024, 1024, 1024, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=1024,
        dropout_ratio=0.2,
        num_classes=2,
        out_channels=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='FocalTverskyLoss', loss_name='loss_focal', loss_weight=1.0, #class_weight=[0.1, 1.0]
             )),
    # auxiliary_head=dict(
    #     type='FCNHead',
    #     in_channels=1024,
    #     in_index=2,
    #     channels=256,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=2,
    #     out_channels=2,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(type='FocalTverskyLoss', loss_name='loss_focal', loss_weight=1.0, #class_weight=[0.1, 1.0]
    #          )),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
