from jseg.utils.registry import OPTIMS
import jittor as jt
from jittor import optim


class Optimizer(object):
    def parameters(self):
        data = {}
        for k, d in self.__dict__.items():
            if k == "param_groups":
                continue
            data[k] = d
        return data

    def load_parameters(self, data):
        if isinstance(data, dict):
            for k, d in data.items():
                if k in self.__dict__:
                    self.__dict__[k] = d

    def cur_lr(self):
        return self.param_groups[0].get("lr", self.lr)


@OPTIMS.register_module()
class SGD(optim.SGD, Optimizer):
    def __init__(self,
                 params,
                 lr,
                 momentum=0,
                 weight_decay=0,
                 dampening=0,
                 nesterov=False,
                 grad_clip=None):
        super(SGD, self).__init__(params, lr, momentum, weight_decay,
                                  dampening, nesterov)
        self.grad_clip = grad_clip

    def pre_step(self, loss, retain_graph=False):
        super(SGD, self).pre_step(loss)
        if self.grad_clip is not None:
            self.clip_grad_norm(**self.grad_clip)


@OPTIMS.register_module()
class GradMutilpySGD(optim.SGD, Optimizer):
    def __init__(self, grad_clip=None, **kwargs):
        super(GradMutilpySGD, self).__init__(**kwargs)
        self.grad_clip = grad_clip

    def step(self, loss):
        if loss is not None:
            self.pre_step(loss)
        if self.grad_clip is not None:
            self.clip_grad_norm(**self.grad_clip)
        for pg in self.param_groups:
            # get arguments from each param_groups
            lr = pg.get("lr", self.lr)
            momentum = pg.get("momentum", self.momentum)
            weight_decay = pg.get("weight_decay", self.weight_decay)
            dampening = pg.get("dampening", self.dampening)
            nesterov = pg.get("nesterov", self.nesterov)

            m = pg.get("grad_mutilpy", 1)
            # optimize main body
            for p, g, v in zip(pg["params"], pg["grads"], pg["values"]):
                if p.is_stop_grad():
                    continue
                dp = p * weight_decay + g * m
                v.update(momentum * v + dp * (1 - dampening))
                if nesterov:
                    p.update(p - (dp + momentum * v) * lr)
                else:
                    p.update(p - v * lr)
        self.zero_grad()


@OPTIMS.register_module()
class AdamW(optim.AdamW, Optimizer):
    def __init__(self,
                 params,
                 lr,
                 eps=1e-8,
                 betas=(0.9, 0.999),
                 weight_decay=0):
        super(AdamW, self).__init__(params, lr, eps, betas, weight_decay)

    def pre_step(self, loss, retain_graph=False):
        super(AdamW, self).pre_step(loss)


@OPTIMS.register_module()
class CustomAdamW(AdamW, Optimizer):
    def __init__(self,
                 params,
                 lr,
                 eps=1e-8,
                 betas=(0.9, 0.999),
                 weight_decay=0):
        super(CustomAdamW, self).__init__(params, lr, eps, betas, weight_decay)

    def step(self, loss):
        if loss is not None:
            self.pre_step(loss)

        n = float(self.n_step)
        for pg in self.param_groups:
            # get arguments from each param_groups
            lr = pg.get("lr", self.lr) * pg.get("lr_mult", 1)
            eps = pg.get("eps", self.eps)
            weight_decay = pg.get("weight_decay", self.weight_decay) * pg.get(
                "decay_mult", 1)
            b0, b1 = pg.get("betas", self.betas)
            for p, g, v, m in zip(pg["params"], pg["grads"], pg["values"],
                                  pg["m"]):
                if p.is_stop_grad():
                    continue
                p.update(p * (1 - lr * weight_decay))
                bias_correction1 = 1 - b0**n
                bias_correction2 = 1 - b1**n
                m.update(b0 * m + (1 - b0) * g)  # exp_avg
                v.update(b1 * v + (1 - b1) * g * g)  # exp_avg_sq
                denom = jt.sqrt(v) / jt.sqrt(bias_correction2) + eps
                step_size = lr / bias_correction1
                p.update(p - step_size * m / denom)
        self.post_step()
