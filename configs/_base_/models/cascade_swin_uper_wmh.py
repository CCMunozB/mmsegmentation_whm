norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth'
data_preprocessor = dict(
    type='SegDataPreProcessor',
    #mean=[123.675, 116.28, 103.53],
    #std=[58.395, 57.12, 57.375],
    #bgr_to_rgb=True,
    pad_val=0,
    size=(224,224),
    seg_pad_val=255)
model = dict(
    type='CascadeEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    num_stages=2,
    backbone=dict(
        type='SwinTransformer',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        pretrain_img_size=224,
        embed_dims=192,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg),
    decode_head=[dict(
        type='UPerHead',
        in_channels=[192, 384, 768, 1536],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.,
        num_classes=5,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
        dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, class_weight=[0.3, 1.0, 0., 0., 0.]
             ),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0, class_weight=[0.3, 1.0, 0., 0., 0.]
             )
        ]),
                 
                 dict(
        type='UPerHead',
        in_channels=[192, 384, 768, 1536],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.,
        num_classes=5,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
        dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, class_weight=[0.3, 0., 1., 0., 0.]
             ),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0, class_weight=[0.3, 0., 1., 0., 0.]
             )
        ])],
    
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.3,
        num_classes=5,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
        dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=0.4, class_weight=[0.3, 1., 1., 1., 1.]
             ),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.2, class_weight=[0.3, 1., 1., 1., 1.]
             )
        ]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))