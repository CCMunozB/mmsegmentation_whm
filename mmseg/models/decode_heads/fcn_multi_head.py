# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from typing import List

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
from mmseg.utils import ConfigType, SampleList
from ..losses import accuracy
from ..utils import resize
from torch import Tensor


@MODELS.register_module()
class FCNMultiHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 out_images=2,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.out_images = out_images
        super().__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        assert out_images>= 1, \
            f'out_channels must be 1 or more, actual value {out_images}'
            
        
        if out_images == 1:
            convs = []
            #Initial convolution
            convs.append(
                ConvModule(
                    self.in_channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            #Add extra convolution
            for i in range(num_convs - 1):
                convs.append(
                    ConvModule(
                        self.channels,
                        self.channels,
                        kernel_size=kernel_size,
                        padding=conv_padding,
                        dilation=dilation,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            if num_convs == 0:
                self.convs = nn.Identity()
            else:
                self.convs = nn.Sequential(*convs)
            if self.concat_input:
                self.conv_cat = ConvModule(
                    self.in_channels + self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)
        if out_images > 1:
            out_convs = []
            for _ in range(out_images):
                convs = []
                #Initial convolution
                convs.append(
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        kernel_size=kernel_size,
                        padding=conv_padding,
                        dilation=dilation,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                #Add extra convolution
                for i in range(num_convs - 1):
                    convs.append(
                        ConvModule(
                            self.channels,
                            self.channels,
                            kernel_size=kernel_size,
                            padding=conv_padding,
                            dilation=dilation,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg))
                if num_convs == 0:
                    out_convs.append(nn.Identity())
                else:
                    out_convs.append(nn.Sequential(*convs))
                if self.concat_input:
                    out_convs.append(ConvModule(
                        self.in_channels + self.channels,
                        self.channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            self.out_convs = out_convs
            
        self.conv_seg2 = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)
                

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
                
        """
        
        for val in inputs:
            print(val.shape)
        x = self._transform_inputs(inputs)
        if self.out_images == 1:
            feats = self.convs(x)
            if self.concat_input:
                feats = self.conv_cat(torch.cat([x, feats], dim=1))
        else:
            feats = []
            for conv_modules in self.out_convs:
                feats.append(conv_modules(x))
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
    
    def cls_seg(self, feat):
        """Classify each pixel for all outputs."""
        if self.dropout is not None:
            feat1 = self.dropout(feat[0])
            feat2 = self.dropout(feat[1])
        output = self.conv_seg(feat1)
        output_full = self.conv_seg2(feat2)
        return [output, output_full]
    
    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples if data_sample.sample == 1
        ]
        gt_semantic_segs2 = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples if data_sample.sample == 2
        ]
        
        stack = [torch.stack(gt_semantic_segs, dim=0), torch.stack(gt_semantic_segs2, dim=0)]
        return stack
    
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

        #Only avaible por 2 outputs -- may fix later
        
        seg_label = self._stack_batch_gt(batch_data_samples)
        #Here we have a list with both results ---> Solve for both solutions
        loss = dict()
        #seg_logits comes from forward -> two inputs, so we nedd to diferentiate to all input/output
        seg_logits1 = resize(
            input=seg_logits[1],
            size=seg_label[0].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        
        #Second one is full concatenation
        seg_logits2 = resize(
            input=seg_logits[0],
            size=seg_label[0].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        #Both Outputs have the same dimensions, therefore, the reshape on the first one.
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None

            
        #Get both segmentation labels, 0 is normal
        seg_label1 = seg_label[0].squeeze(1)
        seg_label2 = seg_label[1].squeeze(1)
        
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                #Both losses are added a 50/50 proportion
                loss[loss_decode.loss_name] = 0.5*loss_decode(
                    seg_logits1,
                    seg_label1,
                    weight=seg_weight,
                    ignore_index=self.ignore_index) + 0.5*loss_decode(
                    seg_logits2,
                    seg_label2,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += 0.5*loss_decode(
                    seg_logits1,
                    seg_label1,
                    weight=seg_weight,
                    ignore_index=self.ignore_index) + 0.5*loss_decode(
                    seg_logits2,
                    seg_label2,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        #Change Accuracy ofr multi outputs
        loss['acc_seg'] = accuracy(
            seg_logits1, seg_label1, ignore_index=self.ignore_index)
        return loss
    
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
            input=seg_logits[0],
            size=batch_img_metas[0]['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners)
        return seg_logits
