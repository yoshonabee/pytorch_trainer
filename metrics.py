class BaseMetric(object):
    name = 'base_metric'

    def __init__(self, eps=10e-7):
        self.eps = eps

    def __call__(self, pred, target):
        raise NotImplementedError()

class MAPE(BaseMetric):
    name = 'mape'

    def __init__(self, *args, **kwargs):
        super(MAPE, self).__init__(*args, **kwargs)

    def __call__(self, pred, target):
        loss = (pred - target).abs()
        score = loss / (target + self.eps)

        return score.mean().item()

class Accuracy(BaseMetric):
    name= 'accuracy'

    def __init__(self, *args, **kwargs):
        super(Accuracy, self).__init__(*args, **kwargs)

    def __call__(self, pred, target):
        return (pred.max(-1)[1] == target).float().mean().item()

class Precision(BaseMetric):
    name= 'precision'

    def __init__(self, *args, **kwargs):
        super(Precision, self).__init__(*args, **kwargs)

    def __call__(self, pred, target):
        tp, fn, fp, tn = classification_measurements(pred, target)
        return tp / (tp + fp + self.eps)

class Recall(BaseMetric):
    name= 'recall'

    def __init__(self, *args, **kwargs):
        super(Recall, self).__init__(*args, **kwargs)

    def __call__(self, pred, target):
        tp, fn, fp, tn = classification_measurements(pred, target)
        return tp / (tp + fn + self.eps)

class F1(BaseMetric):
    name= 'f1-score'

    def __init__(self, *args, **kwargs):
        super(F1, self).__init__(*args, **kwargs)

    def __call__(self, pred, target):
        tp, fn, fp, tn = classification_measurements(pred, target)
        return 2 * tp / (2 * tp + fn + fp + self.eps)

def classification_measurements(pred, target):
    pred = pred.detach().cpu().max(-1)[1].view(-1).numpy()
    target = target.detach().cpu().view(-1).numpy()

    tp, fn, fp, tn = 0, 0, 0, 0

    for p, t in zip(pred, target):
        if t:
            if p:
                tp += 1
            else:
                fn += 1
        else:
            if p:
                fp += 1
            else:
                tn += 1
    
    return tp, fn, fp, tn


