# Exploring Regional Clues in CLIP for Zero-Shot Semantic Segmentation (CVPR 2024)


The repository contains official Jittor implementations of the paper: Exploring Regional Clues in CLIP for Zero-Shot Semantic Segmentation. 

The paper is in [**Here**]().

**Notes**: CLIP-ViT-B-16 Pre-trained models can be found in [there](https://bhpan.buaa.edu.cn/link/AA95601A0FBCA5403485078A0160952FEC)

## Pretrained models

|     Dataset     |   Setting    |  pAcc | mIoU(S) | mIoU(U) | hIoU |                           Model Zoo                           |
| :-------------: | :---------:  | :---: | :-----: | :-----: | :--: |  :----------------------------------------------------------: |
| PASCAL VOC 2012 |  Inductive   |  95.8 |   92.8  |   84.4  | 88.4 | [[Drive](https://bhpan.buaa.edu.cn/link/AA10306CBF37904DDCB835F3BE2D7B1C15)] |
| PASCAL VOC 2012 | Transductive |  97.0 |   93.9  |   92.2  | 93.0 | [[Drive](https://bhpan.buaa.edu.cn/link/AAE085202961AF45CD957E9F98BB7449FB)] |
| PASCAL VOC 2012 |    Fully     |  97.1 |   94.1  |   93.4  | 93.7 | [[Drive](https://bhpan.buaa.edu.cn/link/AAA98108D9C3DD408C82B42EC206DD95DD)] |
| COCO Stuff 164K |  Inductive   |  63.1 |   40.9  |   41.6  | 41.2 | [[Drive](https://bhpan.buaa.edu.cn/link/AA12C2F1BBA0804EC6820A8CB160062091)]|
| COCO Stuff 164K | Transductive |  69.9 |   42.0  |   60.8  | 49.7 | [[Drive](https://bhpan.buaa.edu.cn/link/AA492DE7FE832E43D299C221931127CB1D)]|
| COCO Stuff 164K |    Fully     |  70.8 |   42.9  |   64.1  | 51.4 | [[Drive](https://bhpan.buaa.edu.cn/link/AACE6B7E6F7DED41FDA09AF4CB308F4E2A)] |
