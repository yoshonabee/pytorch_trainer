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