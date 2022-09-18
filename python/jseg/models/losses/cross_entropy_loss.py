import jittor as jt
from jittor import nn

from jseg.utils.registry import LOSSES
from .utils import weight_reduce_loss


def cross_entropy_loss(output,
                       target,
                       weight=None,
                       ignore_index=None,
                       reduction='sum'):
    if len(output.shape) == 4:
        c_dim = output.shape[1]
        output = output.transpose((0, 2, 3, 1))
        output = output.reshape((-1, c_dim))

    target = target.reshape((-1, ))
    target_weight = jt.ones(target.shape[0], dtype='float32')
    if weight is not None:
        target_weight = weight[target]
    if ignore_index is not None:
        target_weight = jt.ternary(target == ignore_index,
                                   jt.array(0).broadcast(target_weight),
                                   target_weight)

    target = target.broadcast(output, [1])
    target = target.index(1) == target

    output = output - output.max([1], keepdims=True)
    logsum = output.exp().sum(1).log()
    loss = (logsum - (output * target).sum(1))
    if reduction == 'sum':
        return (loss * target_weight).sum() / target_weight.sum()
    elif reduction == 'mean':
        return (loss * target_weight).mean() / target_weight.mean()
    else:
        loss[target_weight == 0] = 0
        return loss


def cross_entropy(pred,
                  label,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100):
    loss = cross_entropy_loss(pred,
                              label,
                              weight=class_weight,
                              reduction='none',
                              ignore_index=ignore_index)

    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss,
                              weight=weight,
                              reduction=reduction,
                              avg_factor=avg_factor)

    return loss


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        self.cls_criterion = cross_entropy

    def execute(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = jt.Var(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls
