# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class ShiftDataset(BaseSegDataset):
    # """
    # The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    # fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    # """
    # METAINFO = dict(
    #     classes=('background', 'teeth', 'k00', 'k01', 'k02', 'k03', 'k04',
    #              'k05', 'k06', 'k07', 'k08',
    #              'k09', 'k10', 'dental restoration', 'root canal treatment', 'prosthesis', 'implant'),
    #     palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    #              [190, 153, 153], [153, 153, 153], [250, 170,
    #                                                 30], [220, 220, 0],
    #              [107, 142, 35], [152, 251, 152], [70, 130, 180],
    #              [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
    #              [0, 60, 100], [0, 80, 100]])

    # def __init__(self,
    #              img_suffix='.png',
    #              seg_map_suffix='_mask.png',
    #              **kwargs) -> None:
    #     super().__init__(
    #         img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = dict(

        classes=('background', 'building', 'fence', 'other', ' pedestrian', ' pole', 'road line', 'road', 'sidewalk',
                 'vegetation', 'vehicle', 'wall', 'traffic sign', 'sky', 'ground', 'bridge', 'rail track', 'guard rail',
                 'traffic light', 'static', 'dynamic', 'water', 'terrain'
                ),
        # palette is a list of color tuples, which is used for visualization.
        palette=[[0,0,0], ( 70, 70, 70),  (100, 40, 40), ( 55, 90, 80), (220, 20, 60), 
                (153, 153, 153), (157, 234, 50), (128, 64, 128), (244, 35, 232), (107, 142, 35),
                ( 0, 0, 142), (102, 102, 156), (220, 220, 0), 	( 70, 130, 180), ( 81, 0, 81), (150, 100, 100),
                (230, 150, 140), (180, 165, 180), (250, 170, 30), (110, 190, 160), (170, 120, 50),
                ( 45, 60, 150), (145, 170, 100)
                 ]
        
        )

    def __init__(self,
                #  reduce_zero_label=True,
                 img_suffix='img_front.jpg',
                 seg_map_suffix='semseg_front.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            # reduce_zero_label=reduce_zero_label,
            seg_map_suffix=seg_map_suffix,
            **kwargs)