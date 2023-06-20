from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class WMHDataset(BaseSegDataset):
    """White Matter Hyperintensities Segmentation Challenge dataset.

    In segmentation map annotation, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.tif'.
    """
    METAINFO = dict(
        classes=('background', 'WHM', 'Other'),
        palette=[[255, 255, 255], [255, 0, 0], [255, 255, 0]])

    def __init__(self,
                 img_suffix='.nii.gz',
                 seg_map_suffix='.nii.gz',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)