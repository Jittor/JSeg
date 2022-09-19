from jseg.utils.registry import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class iSAIDDataset(CustomDataset):
    CLASSES = ('background', 'ship', 'store_tank', 'baseball_diamond',
               'tennis_court', 'basketball_court', 'Ground_Track_Field',
               'Bridge', 'Large_Vehicle', 'Small_Vehicle', 'Helicopter',
               'Swimming_pool', 'Roundabout', 'Soccer_ball_field', 'plane',
               'Harbor')
    PALETTE = [[0, 0, 0], [0, 0, 63], [0, 63, 63], [0, 63, 0], [0, 63, 127],
               [0, 63, 191], [0, 63, 255], [0, 127, 63], [0, 127, 127],
               [0, 0, 127], [0, 0, 191], [0, 0, 255], [0, 191, 127],
               [0, 127, 191], [0, 127, 255], [0, 100, 155]]

    def __init__(self, **kwargs):
        super(iSAIDDataset,
              self).__init__(img_suffix='.png',
                             seg_map_suffix='_instance_color_RGB.png',
                             **kwargs)
