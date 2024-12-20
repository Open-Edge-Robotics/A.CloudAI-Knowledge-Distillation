# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class ToothDataset(BaseSegDataset):
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

        classes=('background', 'teeth', 'K01', 'K02', 'K04',
                'K05', 'K08.1', 'K08.3', 'K09',
                'Dental Restoratives&Dental Prosthesis' , 'root canal treatment', 'implant',
                ),
        # palette is a list of color tuples, which is used for visualization.
        palette=[[0,0,0], [128, 64, 128],  [244, 35, 232], [70, 70, 70], [102, 102, 156], 
                 [190, 153, 153], [153, 153, 153], [250, 170,30], [152, 251, 152], 
                 [220, 20, 60], [25, 110, 50], [0, 0, 70],
                 ]
        
        )

    def __init__(self,
                #  reduce_zero_label=True,
                #  img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_mask.png',
                 **kwargs) -> None:
        super().__init__(
            # img_suffix=img_suffix,
            # reduce_zero_label=reduce_zero_label,
            seg_map_suffix=seg_map_suffix, **kwargs)