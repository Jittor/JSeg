from jseg.utils.registry import MODELS


@MODELS.register_module()
def CustomPrameterGroupsGenerator(named_params, model, custom_keys={}, logger=None):
    def get_custom_parameter_groups(name):
        for ck in custom_keys.keys():
            if ck in name:
                return custom_keys[ck]
        return None

    normal_group_list = []
    custom_group_list = []

    for p in named_params:
        name, param = p
        custom_group = get_custom_parameter_groups(name)
        if custom_group is not None:
            tmp = {}
            tmp['params'] = [param]
            for i in custom_group.keys():
                tmp[i] = custom_group.get(i)
            custom_group_list.append(tmp)
            continue
        normal_group_list.append({'params': [param]})
    return normal_group_list + custom_group_list
