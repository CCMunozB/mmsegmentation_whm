# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth'
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
        type='SwinTransformer',
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
    decode_head=dict(
            type='RemosHead',
            in_channels=[192, 384, 768, 1536],
            in_index=[0, 1, 2, 3],
            channels=224,
            dropout_ratio=0.2,
            num_classes=2,
            out_channels=2,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=[
        dict(type='LogCoshDiceLoss', loss_name='loss_cosh', loss_weight=0.8, class_weight=[0.001, 1.0])
        ]),
     auxiliary_head=dict(
         type='FCNHead',
         in_index=2,
         in_channels=768,
         channels=224,
         num_convs=2,
         concat_input=False,
         dropout_ratio=0.2,
         num_classes=2,
         out_channels=2,
         norm_cfg=norm_cfg,
         align_corners=False,
         loss_decode=[
         dict(type='LogCoshDiceLoss', loss_name='loss_cosh', loss_weight=0.2, class_weight=[0.001, 1.0])
         ]),
    # model training and testing settings#     in_channels=768,
    #
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
