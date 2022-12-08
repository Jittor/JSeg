from jseg.utils.registry import MODELS
import json


def get_layer_id_for_convnext(var_name, max_layer_id):
    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.downsample_layers'):
        stage_id = int(var_name.split('.')[2])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3
        elif stage_id == 3:
            layer_id = max_layer_id
        return layer_id
    elif var_name.startswith('backbone.stages'):
        stage_id = int(var_name.split('.')[2])
        block_id = int(var_name.split('.')[3])
        if stage_id == 0:
            layer_id = 1
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3 + block_id // 3
        elif stage_id == 3:
            layer_id = max_layer_id
        return layer_id
    else:
        return max_layer_id + 1


def get_stage_id_for_convnext(var_name, max_stage_id):
    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.downsample_layers'):
        return 0
    elif var_name.startswith('backbone.stages'):
        stage_id = int(var_name.split('.')[2])
        return stage_id + 1
    else:
        return max_stage_id - 1


def get_layer_id_for_vit(var_name, max_layer_id):
    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.patch_embed'):
        return 0
    elif var_name.startswith('backbone.layers'):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return max_layer_id - 1


@MODELS.register_module()
def LRDecayParameterGroupsGenerator(named_params,
                                    model,
                                    paramwise_cfg={},
                                    logger=None):
    num_layers = paramwise_cfg.get('num_layers') + 2
    decay_rate = paramwise_cfg.get('decay_rate')
    decay_type = paramwise_cfg.get('decay_type', 'layer_wise')

    parameter_groups = {}
    normal_group_list = []
    custom_group_list = []
    for p in named_params:
        name, param = p
        if not param.requires_grad:
            normal_group_list.append({'params': [param]})
            continue
        if len(param.shape) == 1 or name.endswith('.bias') or name in (
                'pos_embed', 'cls_token'):
            group_name = 'no_decay'
            decay_mult = 0.
        else:
            group_name = 'decay'
            decay_mult = 1
        if 'layer_wise' in decay_type:
            if 'ConvNeXt' in model.backbone.__class__.__name__:
                layer_id = get_layer_id_for_convnext(
                    name, paramwise_cfg.get('num_layers'))
            elif 'BEiT' in model.backbone.__class__.__name__ or \
                    'MAE' in model.backbone.__class__.__name__:
                layer_id = get_layer_id_for_vit(name, num_layers)
            else:
                raise NotImplementedError()
        elif decay_type == 'stage_wise':
            if 'ConvNeXt' in model.backbone.__class__.__name__:
                layer_id = get_stage_id_for_convnext(name, num_layers)
            else:
                raise NotImplementedError()
        scale = decay_rate**(num_layers - layer_id - 1)
        group_name = f'layer_{layer_id}_{group_name}'
        if group_name not in parameter_groups.keys():
            parameter_groups[group_name] = {
                'decay_mult': decay_mult,
                'lr_mult': scale,
                'param_names': []
            }
        parameter_groups[group_name]['param_names'].append(name)
        custom_group_list.append({
            'decay_mult': decay_mult,
            'params': [param],
            'lr_mult': scale,
        })
    logger.log({'parameter_groups': json.dumps(parameter_groups, indent=2)})
    return normal_group_list + custom_group_list
