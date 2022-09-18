import jittor as jt
import time
import warnings
import numpy as np
import random
import os
import glob
from functools import partial
from six.moves import map, zip
import numpy
import tempfile
from pathlib import Path
import os.path as osp
from collections import abc


def is_seq_of(seq, expected_type, seq_type=None):
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_tuple_of(seq, expected_type):
    return is_seq_of(seq, expected_type, seq_type=tuple)


def is_list_of(seq, expected_type):
    return is_seq_of(seq, expected_type, seq_type=list)


def list_from_file(filename, prefix='', offset=0, max_num=0, encoding='utf-8'):
    cnt = 0
    item_list = []
    with open(filename, 'r', encoding=encoding) as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if 0 < max_num <= cnt:
                break
            item_list.append(prefix + line.rstrip('\n\r'))
            cnt += 1
    return item_list


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def scandir(dir_path, suffix=None, recursive=False):
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                if suffix is None or rel_path.endswith(suffix):
                    yield rel_path
            elif recursive and os.path.isdir(entry.path):
                # scan recursively if entry.path is a directory
                yield from _scandir(entry.path,
                                    suffix=suffix,
                                    recursive=recursive)

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def to_jt_var(data):
    """
        convert data to jt_array
    """
    def _to_jt_var(data):
        if isinstance(data, (list, tuple)):
            data = [_to_jt_var(d) for d in data]
        elif isinstance(data, dict):
            data = {k: _to_jt_var(d) for k, d in data.items()}
        elif isinstance(data, numpy.ndarray):
            data = jt.array(data)
        elif not isinstance(data, (int, float, str, np.ndarray)):
            raise ValueError(f"{type(data)} is not supported")
        return data

    return _to_jt_var(data)


def sync(data, reduce_mode="mean", to_numpy=True):
    def _sync(data):
        if isinstance(data, (list, tuple)):
            data = [_sync(d) for d in data]
        elif isinstance(data, dict):
            data = {k: _sync(d) for k, d in data.items()}
        elif isinstance(data, jt.Var):
            if jt.in_mpi:
                data = data.mpi_all_reduce(reduce_mode)
            if to_numpy:
                data = data.numpy()
        elif not isinstance(data, (int, float, str, np.ndarray)):
            raise ValueError(f"{type(data)} is not supported")
        return data

    return _sync(data)


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    if data.ndim == 1:
        ret = jt.full((count, ), fill, dtype=data.dtype)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = jt.full(new_size, fill, dtype=data.dtype)
        ret[inds, :] = data
    return ret


def parse_losses(losses):
    _losses = dict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, jt.Var):
            _losses[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            _losses[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    total_loss = sum(_value for _key, _value in _losses.items()
                     if 'loss' in _key)
    return total_loss, _losses


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    jt.seed(seed)


def current_time():
    return time.asctime(time.localtime(time.time()))


def check_file(file, ext=None):
    if file is None:
        return False
    if not os.path.exists(file):
        warnings.warn(f"{file} is not exists")
        return False
    if not os.path.isfile(file):
        warnings.warn(f"{file} must be a file")
        return False
    if ext:
        if not os.path.splitext(file)[1] in ext:
            # warnings.warn(f"the type of {file} must be in {ext}")
            return False
    return True


def build_file(work_dir, prefix):
    work_dir = os.path.abspath(work_dir)
    prefixes = prefix.split("/")
    file_name = prefixes[-1]
    prefix = "/".join(prefixes[:-1])
    if len(prefix) > 0:
        work_dir = os.path.join(work_dir, prefix)
    os.makedirs(work_dir, exist_ok=True)
    file = os.path.join(work_dir, file_name)
    return file


def check_interval(step, step_interval):
    if step is None or step_interval is None:
        return False
    if step and step % step_interval == 0:
        return True
    return False


def check_dir(work_dir):
    os.makedirs(work_dir, exist_ok=True)


def list_files(file_dir):
    if os.path.isfile(file_dir):
        return [file_dir]

    filenames = []
    for f in os.listdir(file_dir):
        ff = os.path.join(file_dir, f)
        if os.path.isfile(ff):
            filenames.append(ff)
        elif os.path.isdir(ff):
            filenames.extend(list_files(ff))

    return filenames


def is_img(f):
    ext = os.path.splitext(f)[1]
    return ext.lower() in [".jpg", ".bmp", ".jpeg", ".png", "tiff"]


def list_images(img_dir):
    img_files = []
    for img_d in img_dir.split(","):
        if len(img_d) == 0:
            continue
        if not os.path.exists(img_d):
            raise f"{img_d} not exists"
        img_d = os.path.abspath(img_d)
        img_files.extend([f for f in list_files(img_d) if is_img(f)])
    return img_files


def search_ckpt(work_dir):
    files = glob.glob(os.path.join(work_dir, "checkpoints/ckpt_*.pkl"))
    if len(files) == 0:
        return None
    files = sorted(files, key=lambda x: int(x.split("_")[-1].split(".pkl")[0]))
    return files[-1]


def add_prefix(inputs, prefix):
    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs


def np2tmp(array, temp_file_name=None):
    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(suffix='.npy',
                                                     delete=False).name
    np.save(temp_file_name, array)
    return temp_file_name
