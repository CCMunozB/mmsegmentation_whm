# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmcv.cnn.bricks.transformer import MultiheadAttention #####################################################################################
from mmcv.cnn import build_norm_layer
from mmseg.registry import MODELS
from ..utils import resize

import warnings
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor

from mmseg.structures import build_pixel_sampler
from mmseg.utils import ConfigType, SampleList
from ..builder import build_loss
from ..losses import accuracy
from ..utils import resize

@MODELS.register_module()
class RemosHead(BaseDecodeHead):
    """
    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', remos=3, remos_weight=[0.125, 0.125, 0.125, 0.625], **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)
        assert remos <= (num_inputs - 1)
        assert remos == (len(remos_weight) - 1)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            
        ##Add MultiheadAtenttion

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        
        self.remos_fusion = []
        for _ in range(remos):
            self.remos_fusion.append(ConvModule(in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=1,
                norm_cfg=self.norm_cfg).cuda())
        
        self.remos_conv_seg = []
        # Dimension example for remos=3 [N/2, N/4, N/8, N]
        for _ in range(remos):
            self.remos_conv_seg.append(nn.Conv2d(self.channels, self.out_channels, kernel_size=1).cuda())
            
        self.remos_weight = remos_weight

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))
            
        remos_out = []

        for idx in range(len(self.remos_fusion)):
            remos_conv = self.remos_fusion[idx](outs[idx])
            remos_out.append(remos_conv)
        #Test 2 conection types full merge output or half merge output
        
        pre_out_full = self.fusion_conv(torch.cat(outs, dim=1))
        
        remos_out_final = [remos_out[0], remos_out[1], remos_out[2], pre_out_full]

        out = self.cls_seg(remos_out_final)

        return out
    
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
    
    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        #Two outputs in an array
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses
    
    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        seg_logits = self.forward(inputs)

        return self.predict_by_feat(seg_logits, batch_img_metas)
    
    
    ## data samples format?
    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)
    
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
    
    
