# reference: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py
from jseg.utils.registry import SCHEDULERS
import math


@SCHEDULERS.register_module()
class WarmUpLR(object):
    def __init__(self,
                 optimizer,
                 warmup_ratio=1.0 / 3,
                 warmup_iters=500,
                 warmup=None):
        self.optimizer = optimizer
        self.warmup_ratio = warmup_ratio
        self.warmup_iters = warmup_iters
        self.warmup = warmup
        self.base_lr = optimizer.lr
        self.base_lr_pg = [
            pg.get("lr", optimizer.lr) for pg in optimizer.param_groups
        ]
        self.step(0, 0)

    def get_warmup_lr(self, lr, cur_iters):
        if self.warmup == 'constant':
            k = self.warmup_ratio
        elif self.warmup == 'linear':
            k = 1 - (1 - cur_iters / self.warmup_iters) * (1 -
                                                           self.warmup_ratio)
        elif self.warmup == 'exp':
            k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
        return k * self.get_lr(lr, cur_iters)

    def get_lr(self, lr, steps):
        return lr

    def _update_lr(self, steps, get_lr_func):
        self.optimizer.lr = get_lr_func(self.base_lr, steps)
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = get_lr_func(self.base_lr_pg[i], steps)

    def step(self, iters, epochs, by_epoch=True):
        if self.warmup is not None:
            if iters >= self.warmup_iters:
                if by_epoch:
                    self._update_lr(epochs, self.get_lr)
                else:
                    self._update_lr(iters, self.get_lr)
            else:
                self._update_lr(iters, self.get_warmup_lr)
        else:
            if by_epoch:
                self._update_lr(epochs, self.get_lr)
            else:
                self._update_lr(iters, self.get_lr)

    def parameters(self):
        return {
            key: value
            for key, value in self.__dict__.items() if key != 'optimizer'
        }

    def load_parameters(self, data):
        if isinstance(data, dict):
            for k, d in data.items():
                if k in self.__dict__:
                    self.__dict__[k] = d


@SCHEDULERS.register_module()
class WarmUpLRGroup(object):
    def __init__(self,
                 optimizer,
                 warmup_ratio=1.0 / 3,
                 warmup_ratio_pg=None,
                 warmup_init_lr_pg=None,
                 warmup_iters=500,
                 warmup=None,
                 warmup_pg=None,
                 warmup_initial_momentum=None):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.base_lr_pg = [
            pg.get("lr", optimizer.lr) for pg in optimizer.param_groups
        ]
        self.warmup = warmup
        self.warmup_ratio = warmup_ratio

        self.warmup_pg = warmup_pg if warmup_pg is not None else [
            warmup for _ in self.base_lr_pg
        ]

        if warmup_init_lr_pg or warmup_ratio_pg:
            assert (warmup_init_lr_pg is None) ^ (warmup_ratio_pg is
                                                  None), 'only one can be set'
            if warmup_init_lr_pg:
                self.warmup_ratio_pg = [
                    warmup_init_lr_pg[i] / self.base_lr_pg[i]
                    for i in range(len(warmup_init_lr_pg))
                ]
            else:
                self.warmup_ratio_pg = warmup_ratio_pg
            assert len(self.warmup_ratio_pg) == len(
                self.base_lr_pg), 'these two must be the same'
        else:
            self.warmup_ratio_pg = [self.warmup_ratio for _ in self.base_lr_pg]

        if warmup_initial_momentum is not None:
            self.warmup_initial_momentum = warmup_initial_momentum
            self.base_mom_pg = [
                pg.get('momentum', -1.) for pg in optimizer.param_groups
            ]
        else:
            self.warmup_initial_momentum = None
        self.warmup_iters = warmup_iters
        self.step(0, 0)

    def get_warmup_lr(self, lr, cur_iters, warmup=None, ratio=None, epochs=0):
        if warmup == 'constant':
            k = ratio
        elif warmup == 'linear':
            # in the linear case, the ratio is adjusted based on current learning rate.
            ratio *= lr / self.get_lr(lr, epochs)
            k = 1 - (1 - cur_iters / self.warmup_iters) * (1 - ratio)
        elif warmup == 'exp':
            k = ratio**(1 - cur_iters / self.warmup_iters)
        return k * lr

    def get_warmup_mom(self, momentum, cur_iters):
        # only supports linear
        return self.warmup_initial_momentum + (
            momentum - self.warmup_initial_momentum) * (cur_iters /
                                                        self.warmup_iters)

    def get_lr(self, lr, steps, warmup=None, ratio=None):
        return lr

    def _update_lr(self, steps, get_lr_func):
        self.optimizer.lr = get_lr_func(self.base_lr, steps)
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = get_lr_func(self.base_lr_pg[i], steps)

    def _update_mom(self, steps, get_mom_func):
        for i, param_group in enumerate(self.optimizer.param_groups):
            if 'momentum' in param_group:
                param_group['momentum'] = get_mom_func(self.base_mom_pg[i],
                                                       steps)

    def _update_warmup_lr(self, steps, get_lr_func, epochs=0):
        self.optimizer.lr = get_lr_func(self.base_lr,
                                        steps,
                                        warmup=self.warmup,
                                        ratio=self.warmup_ratio)
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = get_lr_func(self.base_lr_pg[i],
                                            steps,
                                            warmup=self.warmup_pg[i],
                                            ratio=self.warmup_ratio_pg[i],
                                            epochs=epochs)

    def step(self, iters, epochs, by_epoch=True):
        if self.warmup is not None:
            if iters >= self.warmup_iters:
                if by_epoch:
                    self._update_lr(epochs, self.get_lr)
                else:
                    self._update_lr(iters - self.warmup_iters, self.get_lr)
            else:
                self._update_warmup_lr(iters,
                                       self.get_warmup_lr,
                                       epochs=epochs)
                if self.warmup_initial_momentum:
                    self._update_mom(iters, self.get_warmup_mom)
        else:
            if by_epoch:
                self._update_lr(epochs, self.get_lr)
            else:
                self._update_lr(iters, self.get_lr)

    def parameters(self):
        return {
            key: value
            for key, value in self.__dict__.items() if key != 'optimizer'
        }

    def load_parameters(self, data):
        if isinstance(data, dict):
            for k, d in data.items():
                if k in self.__dict__:
                    self.__dict__[k] = d


@SCHEDULERS.register_module()
class StepLR(WarmUpLR):
    def __init__(self, milestones, gamma=0.1, min_lr=None, **kwargs):
        if isinstance(milestones, list):
            assert all([s > 0 for s in milestones])
        elif isinstance(milestones, int):
            assert milestones > 0
        else:
            raise TypeError('"step" must be a list or integer')
        self.milestones = milestones
        self.gamma = gamma
        self.min_lr = min_lr
        super(StepLR, self).__init__(**kwargs)

    def get_lr(self, base_lr, steps):
        # calculate exponential term
        if isinstance(self.milestones, int):
            exp = steps // self.milestones
        else:
            exp = len(self.milestones)
            for i, s in enumerate(self.milestones):
                if steps < s:
                    exp = i
                    break

        lr = base_lr * (self.gamma**exp)
        if self.min_lr is not None:
            # clip to a minimum value
            lr = max(lr, self.min_lr)
        return lr


@SCHEDULERS.register_module()
class CosineAnnealingLR(WarmUpLR):
    def __init__(self, max_steps, min_lr=None, min_lr_ratio=None, **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.max_steps = max_steps
        super(CosineAnnealingLR, self).__init__(**kwargs)

    def get_lr(self, base_lr, steps):
        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr
        cos_out = math.cos(math.pi * (steps / self.max_steps)) + 1
        lr = target_lr + 0.5 * (base_lr - target_lr) * cos_out
        return lr


@SCHEDULERS.register_module()
class CosineAnnealingLRGroup(WarmUpLRGroup):
    def __init__(self, max_steps, min_lr=None, min_lr_ratio=None, **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.max_steps = max_steps
        super(CosineAnnealingLRGroup, self).__init__(**kwargs)

    def get_lr(self, base_lr, steps):
        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr
        cos_out = math.cos(math.pi * (steps / self.max_steps)) + 1
        lr = target_lr + 0.5 * (base_lr - target_lr) * cos_out
        return lr


@SCHEDULERS.register_module()
class ExpLR(WarmUpLR):
    def __init__(self, gamma, **kwargs):
        self.gamma = gamma
        super(ExpLR, self).__init__(**kwargs)

    def get_lr(self, base_lr, steps):
        return base_lr * self.gamma**steps


@SCHEDULERS.register_module()
class PolyLR(WarmUpLR):
    def __init__(self, max_steps, power=1., min_lr=0., **kwargs):
        self.power = power
        self.min_lr = min_lr
        self.max_steps = max_steps
        super(PolyLR, self).__init__(**kwargs)

    def get_lr(self, base_lr, steps):
        coeff = (1 - steps / self.max_steps)**self.power
        return (base_lr - self.min_lr) * coeff + self.min_lr


@SCHEDULERS.register_module()
class InvLR(WarmUpLR):
    def __init__(self, gamma, power=1., **kwargs):
        self.gamma = gamma
        self.power = power
        super(InvLR, self).__init__(**kwargs)

    def get_lr(self, base_lr, steps):
        return base_lr * (1 + self.gamma * steps)**(-self.power)
