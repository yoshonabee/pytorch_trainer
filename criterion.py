import torch
import torch.nn as nn

class DefaultCriterion(nn.Module):
    CRITERIONS = {
        "mse": nn.MSELoss,
        "l1": nn.L1Loss,
        "mae": nn.L1Loss,
        "crossentropy": nn.CrossEntropyLoss
    }

    def __init__(self, criterion):
        super(DefaultCriterion, self).__init__()
        if criterion is not None:
            self.criterion = self.CRITERIONS[criterion.lower()]()
        else:
            self.criterion = None

        self.device = torch.device("cpu")

    def forward(self, model, data):
        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)

        pred = model(x)
        loss = self.criterion(pred.view(-1, pred.size(-1)), y.view(-1))

        return loss, pred, y

    def to(self, device):
        if self.criterion is not None:
            self.criterion.to(device)
        self.device = device
