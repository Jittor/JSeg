from .custom import CustomDataset
from .isaid import iSAIDDataset
from .ade import ADE20KDataset
from .voc import PascalVOCDataset
from .cityscapes import CityscapesDataset
from .loveda import LoveDADataset
from .isprs import ISPRSDataset
from .potsdam import PotsdamDataset
from .zero_voc12 import ZeroPascalVOCDataset20
from .zero_coco_stuff import ZeroCOCOStuffDataset

__all__ = [
    'CustomDataset', 'iSAIDDataset', 'ADE20KDataset', 'PascalVOCDataset',
    'CityscapesDataset', 'LoveDADataset', 'ISPRSDataset', 'PotsdamDataset',
    'ZeroPascalVOCDataset20', 'ZeroCOCOStuffDataset'
]
