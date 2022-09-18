# How to use configs in JSeg
## Basic usages
### .py configuration files
You can do some easy computation in the .py configuration file:
```python
# cfg.py
import os
exp_id = 1
# path setting
output_path = 'experiments'
root_path = os.path.join(output_path, str(exp_id))
log_path = os.path.join(root_path, 'logs')

# easy calculation
gpus = [0,1,2,3]
n_gpus = len(gpus)
batch_size = 16
base_lr = batch_size * 0.001

# model setting
model = {
    'type': 'Resnet50',
    'return_stages': = ['layer1','layer2','layer3','layer4'],
    'pretrained': True
}
```
You can load .py configuration file as load .yaml configuration file:
```python
# main.py
from jseg.config import init_cfg
init_cfg('cfg.py')
```

Please refer to `[ROOT]/python/jseg/config/config.py` for more details.