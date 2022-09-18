from jseg.utils.general import build_file, current_time
from .registry import HOOKS, build_from_cfg
import time
import os
from tensorboardX import SummaryWriter
from jseg.config import get_cfg


@HOOKS.register_module()
class TextLogger:
    def __init__(self, work_dir):
        save_file = build_file(
            work_dir,
            prefix="textlog/log_" +
            time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + ".txt")
        self.log_file = open(save_file, "a")

    def log(self, data):
        msg = ",".join([f"{k}:{d}" for k, d in data.items()])
        msg = current_time() + msg + "\n"
        self.log_file.write(msg)
        self.log_file.flush()


@HOOKS.register_module()
class TensorboardLogger:
    def __init__(self, work_dir):
        self.cfg = get_cfg()
        tensorboard_dir = os.path.join(work_dir, "tensorboard")
        self.writer = SummaryWriter(tensorboard_dir, flush_secs=10)

    def log(self, data):
        if "iter" in data.keys():
            step = data["iter"]
            for k, d in data.items():
                if k in ["iter", "epoch", "batch_idx", "times", "batch_size"]:
                    continue
                if isinstance(d, str):
                    continue
                self.writer.add_scalar(k, d, global_step=step)


@HOOKS.register_module()
class RunLogger:
    def __init__(self, work_dir, loggers=["TextLogger", "TensorboardLogger"]):
        self.loggers = [
            build_from_cfg(log, HOOKS, work_dir=work_dir) for log in loggers
        ]

    def log(self, data, **kwargs):
        data.update(kwargs)
        data = {
            k: d.item() if hasattr(d, "item") else d
            for k, d in data.items()
        }
        for logger in self.loggers:
            logger.log(data)
        self.print_log(data)

    def get_time(self, s):
        s = int(s)
        days = s // 60 // 60 // 24
        hours = s // 60 // 60 % 24
        minutes = s // 60 % 60
        seconds = s % 60
        return f' [{days}D:{hours}H:{minutes}M:{seconds}S] '

    def print_log(self, msg):
        if isinstance(msg, dict):
            msgs = []
            for k, d in msg.items():
                if (k == "remain_time"):
                    msgs.append(f" {k}:{self.get_time(d)}")
                else:
                    msgs.append(f" {k}:{d:.7f}"
                                if isinstance(d, float) else f" {k}:{d}")
            msg = ",".join(msgs)
        print(current_time(), msg)
