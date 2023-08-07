from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
import mmengine.fileio as fileio
import mmengine
import os.path as osp
from typing import List


@DATASETS.register_module()
class WMHDatasetMulti(BaseSegDataset):
    """White Matter Hyperintensities Segmentation Challenge dataset.

    In segmentation map annotation, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.tif'.
    """
    METAINFO = dict(
        classes=('background', 'WMH',),
        palette=[[120, 120, 120], [180, 120, 120]])

    def __init__(self,
                 img_suffix='.tiff',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        
    def load_data_list(self) -> List[dict]:
        """Load Multiple annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        ann_dir2 = self.data_prefix.get('seg_map_path2', None)
        if osp.isfile(self.ann_file):
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                img_name = line.strip()
                data_info = dict(
                    img_path=osp.join(img_dir, img_name + self.img_suffix))
                if ann_dir is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                    
                if ann_dir2 is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info['seg_map_path2'] = osp.join(ann_dir2, seg_map)
                    
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            for img in fileio.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args):
                data_info = dict(img_path=osp.join(img_dir, img))
                if ann_dir is not None:
                    seg_map = img.replace(self.img_suffix, self.seg_map_suffix)
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                
                if ann_dir2 is not None:
                    seg_map = img.replace(self.img_suffix, self.seg_map_suffix)
                    data_info['seg_map_path2'] = osp.join(ann_dir2, seg_map)
                    
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list