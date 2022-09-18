from jittor import nn
import jittor as jt


def accuracy(pred, target, topk=1, thresh=None):
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    if pred.size(0) == 0:
        accu = [jt.Var(0.) for i in range(len(topk))]
        return accu[0] if return_single else accu
    assert pred.ndim == target.ndim + 1
    assert pred.size(0) == target.size(0)
    assert maxk <= pred.size(1), \
        f'maxk {maxk} exceeds pred dimension {pred.size(1)}'
    pred_value, pred_label = pred.topk(maxk, dim=1)
    # transpose to shape (maxk, N, ...)
    pred_label = pred_label.transpose(1, 0, 2, 3)
    correct = pred_label == target.unsqueeze(0).expand_as(pred_label)
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct = correct & (pred_value > thresh).t()
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdims=True)
        res.append(correct_k.multiply(100.0 / target.numel()))
    return res[0] if return_single else res


class Accuracy(nn.Module):
    def __init__(self, topk=(1, ), thresh=None):

        super().__init__()
        self.topk = topk
        self.thresh = thresh

    def execute(self, pred, target):
        return accuracy(pred, target, self.topk, self.thresh)
