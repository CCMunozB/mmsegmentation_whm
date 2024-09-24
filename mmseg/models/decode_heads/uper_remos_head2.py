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
class UPerRemosHead2(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=None, remos=None, remos_weight=[0.125, 0.125, 0.125, 0.625], **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        
        
        #Laterals
        self.lateral_convs = nn.ModuleList()     
        #ReMOS UpScaling
        self.scaling_convs = nn.ModuleList()
        self.remos_conv_seg = nn.ModuleList()
        last = 0
        for in_channels in self.in_channels:  # skip the top layer
            l_conv = ConvModule(
                int(in_channels/16),
                int(in_channels/16),
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            l_conv2 = ConvModule(
                int(in_channels/16),
                int(in_channels/16),
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv2)
            self.scaling_convs.append(l_conv)
        for in_channels in self.in_channels[::-1]:  # skip the top layer
            r_conv = ConvModule(
                int(in_channels/16) + last,
                self.out_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            last = int(in_channels/16)
            self.remos_conv_seg.append(r_conv)
        # Dimension example for remos=3 [N/2, N/4, N/8, N]
        self.conv_seg = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
            
        self.remos_weight = remos_weight

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
        resize_input = []
        for var in inputs:
            shape = var.shape
            resize_input.append(var.resize(shape[0], int(shape[1]/16), shape[2]*4, shape[3]*4))
        
        pooling = nn.UpsamplingNearest2d(scale_factor=2)
        
        laterals = [
            lateral_conv(resize_input[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        #First pool
        conv = self.scaling_convs[3](laterals[3])
        pool = pooling(conv)
        
        #Uppath
        Up_inputs = [conv]
        for i,orig in enumerate(laterals[::-1]):
            if i == 0:
                pass
            else:
                new_input = torch.concat([pool, orig], dim=1)
                conv = self.scaling_convs[3-i](orig)
                pool = pooling(conv)
                Up_inputs.append(new_input)

        return Up_inputs

    
    def cls_seg(self, feat):
        """Classify each pixel for all outputs."""
        if self.dropout is not None:
            for idx in range(len(feat)):
                feat[idx]= self.dropout(feat[idx])          
        outputs = []
        for idx in range(len(self.remos_conv_seg)):
            outputs.append(self.remos_conv_seg[idx](feat[idx]))
        
        return outputs
    
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
        for _ in range(len(self.remos_conv_seg)- 1):
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
            
        pool_seg_label[-1] = pool_seg_label[-1].squeeze(1)


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
        loss['acc_seg'] = accuracy(seg_logits_list[-1], pool_seg_label[-1], ignore_index=self.ignore_index)
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
