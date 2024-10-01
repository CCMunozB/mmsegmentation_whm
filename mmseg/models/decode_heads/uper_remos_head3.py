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
import torch.nn.functional as F

from mmseg.structures import build_pixel_sampler
from mmseg.utils import ConfigType, SampleList
from ..builder import build_loss
from ..losses import accuracy
from ..utils import resize


def remos_dice_loss(pred, target, smooth=0.0001, exponent=1, **kwards):
    pred = F.softmax(pred, dim=1)
    num_classes = pred.shape[1]
    target = F.one_hot(
        torch.clamp(target.long(), 0, num_classes - 1),
        num_classes=num_classes)
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)

    num = torch.sum(torch.mul(pred, target), dim=1) * 2 + smooth
    den = torch.sum(pred.pow(exponent) + target.pow(exponent), dim=1) + smooth
    
    dice = 1 - num / den

    return dice

@MODELS.register_module()
class UPerRemosHead3(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, remos_weight=[0.25, 0.25, 0.25, 0.25], **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        

        # Expansion Module + Segmentation
        self.remos_convs = nn.ModuleList()
        last_channel = 0
        self.remos_conv_seg = []
        for in_channels in self.in_channels[::-1]:  # skip the top layer
            fpn_conv = ConvModule(
                int(in_channels/16 + last_channel/16),
                int(in_channels/16),
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.remos_convs.append(fpn_conv)
            last_channel = in_channels
            
        for i,in_channels in enumerate(self.in_channels[::-1]):  # skip the top layer
            if i == 3:
                self.conv_seg = nn.Conv2d(int(in_channels/16), 
                          self.out_channels, 
                          kernel_size=1).cuda()
            else:
                self.remos_conv_seg.append(
                nn.Conv2d(int(in_channels/16), 
                          self.out_channels, 
                          kernel_size=1).cuda())
                
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

        resized_inputs = [
            lat.view(-1, int(lat.shape[1]/16), int(lat.shape[2]*4), int(lat.shape[3]*4))
            for lat in inputs
        ][::-1]
        up = nn.UpsamplingNearest2d(scale_factor=2)
        remos_layer = []
        for i, remos_conv in enumerate(self.remos_convs):
            if i == 0:
                res = remos_conv(resized_inputs[i])
            else:
                res = remos_conv(
                    torch.concat(
                        (up(res),resized_inputs[i]), 1)
                    )
            remos_layer.append(res)  

        return remos_layer
    
    def cls_seg(self, feat):
        """Classify each pixel for all outputs."""
        
        if self.dropout is not None:
            for idx in range(len(feat)):
                feat[idx]= self.dropout(feat[idx]) 
                         
        outputs = []
        
        for idx in range(len(self.remos_conv_seg)):
            outputs.append(self.remos_conv_seg[idx](feat[idx]))
            
        outputs.append(self.conv_seg(feat[-1]))

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
        pool_seg_label = [seg_label]
        for _ in range(len(self.remos_conv_seg)):
            pool_seg = pooling(pool_seg)
            pool_seg_label.append(pool_seg.to(torch.uint8))
        
        pool_seg_label = pool_seg_label[::-1]

        loss = dict()
        
        #Pool seg label hast 4 outputs dims [N/2, N/4, N/8, N], should resize for each one.
        seg_logits_list = []

        # for idx in range(len(pool_seg_label)):
        #     seg_logits_list.append(resize(
        #         input=seg_logits[idx],
        #         size=pool_seg_label[idx].shape[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners))
            
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
                loss[loss_decode.loss_name] = sum(list(map(torch.mul,self.remos_weight,list(map(loss_decode,seg_logits,pool_seg_label)))))
            else:
                loss[loss_decode.loss_name] += sum(list(map(torch.mul,self.remos_weight,list(map(loss_decode,seg_logits,pool_seg_label)))))

        #Change Accuracy ofr multi outputs
        #loss['acc_seg'] = accuracy(seg_logits_list[-1], pool_seg_label[-1], ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logits[-1], pool_seg_label[-1], ignore_index=self.ignore_index)
        
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
