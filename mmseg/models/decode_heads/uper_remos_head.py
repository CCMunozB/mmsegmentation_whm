# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead
from .psp_head import PPM
from typing import List, Tuple

from torch import Tensor

from mmseg.structures import build_pixel_sampler
from mmseg.utils import ConfigType, SampleList
from ..builder import build_loss
from ..losses import accuracy
from ..utils import resize

@MODELS.register_module()
class UPerRemosHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), remos=3, remos_weight=[0.125, 0.125, 0.125, 0.625], **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        self.remos_conv_seg = []
        # Dimension example for remos=3 [N/2, N/4, N/8, N]
        for _ in range(remos):
            self.remos_conv_seg.append(nn.Conv2d(self.channels, self.out_channels, kernel_size=1).cuda())
            
        self.remos_weight = remos_weight

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        remos_layers = laterals.copy()
        laterals.append(self.psp_forward(inputs))     
        #Add here layer of learning to spread from top-down path
        

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        remos_layers.append(feats)
        return remos_layers
    
    def cls_seg(self, feat):
        """Classify each pixel for all outputs."""
        if feat is not None:
            if self.dropout is not None:
                for idx in range(len(feat)):
                    feat[idx]= self.dropout(feat[idx])          
            output_full = self.conv_seg(feat[-1])
            outputs = []
            for idx in range(len(self.remos_conv_seg)):
                outputs.append(self.remos_conv_seg[idx](feat[idx]))
            outputs.append(output_full)
        else:
            outputs = [None, None, None, None]
        
        return  outputs
    
    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        
        #To do: Add for non pool feature
        pooling = nn.MaxPool2d(2, stride=2)
        pool_seg = seg_label.to(torch.float64)
        pool_seg_label = []
        for _ in range(len(self.remos_conv_seg)):
            pool_seg = pooling(pool_seg)
            pool_seg_label.append(pool_seg.to(torch.uint8))
        pool_seg_label.append(seg_label)
        

        loss = dict()
        
        #Pool seg label hast 4 outputs dims [N/2, N/4, N/8, N], should resize for each one.
        seg_logits_list = []

        for idx in range(len(pool_seg_label)):
            seg_logits_list.append(resize(
                input=seg_logits[idx],
                size=pool_seg_label[idx].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners))
            
        pool_seg_label = [label.squeeze(1) for label in pool_seg_label]

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None

        #Last in the array is full output

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                #Both losses are added a 50/50 proportion
                loss[loss_decode.loss_name] = sum(list(map(torch.mul,self.remos_weight,list(map(loss_decode,seg_logits_list,pool_seg_label)))))
            else:
                loss[loss_decode.loss_name] += sum(list(map(torch.mul,self.remos_weight,list(map(loss_decode,seg_logits_list,pool_seg_label)))))

        #Change Accuracy ofr multi outputs
        loss['acc_seg'] = accuracy(seg_logits_list[-1], pool_seg_label[-1], ignore_index=self.ignore_index, topk=2)
        return loss

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
    
    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """

        seg_logits = resize(
            input=seg_logits[-1],
            size=batch_img_metas[0]['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners)
        return seg_logits
