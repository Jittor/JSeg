import functools
import warnings


def get_enum(reduction: str) -> int:
    if reduction == 'none':
        ret = 0
    elif reduction == 'mean':
        ret = 1
    elif reduction == 'elementwise_mean':
        warnings.warn(
            "reduction='elementwise_mean' is deprecated, please use reduction='mean' instead."
        )
        ret = 1
    elif reduction == 'sum':
        ret = 2
    else:
        ret = -1  # TODO: remove once JIT exceptions support control flow
        raise ValueError(
            "{} is not a valid value for reduction".format(reduction))
    return ret


def reduce_loss(loss, reduction):
    reduction_enum = get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.ndim == loss.ndim
        if weight.ndim > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper
