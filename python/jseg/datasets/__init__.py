from .custom import CustomDataset
from .isaid import iSAIDDataset
from .ade import ADE20KDataset
from .voc import PascalVOCDataset
from .cityscapes import CityscapesDataset

__all__ = [
    'CustomDataset', 'iSAIDDataset', 'ADE20KDataset', 'PascalVOCDataset',
    'CityscapesDataset'
]
