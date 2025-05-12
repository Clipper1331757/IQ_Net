import torch
import torch.nn as nn


class HuberMRELoss(nn.Module):
    def __init__(self, delta=1.0, alpha=0.5, epsilon=1e-8):
        super(HuberMRELoss, self).__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        huber_loss = self.huber(y_pred, y_true)
        mre_loss = torch.mean(torch.abs(y_pred - y_true) / (torch.abs(y_true) + self.epsilon))

        return self.alpha * huber_loss + (1 - self.alpha) * mre_loss


class LogCoshMRELoss(nn.Module):
    def __init__(self, alpha=0.5, epsilon=1e-8):
        super(LogCoshMRELoss, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # Log-Cosh Loss
        log_cosh_loss = torch.log(torch.cosh(y_pred - y_true))

        # MRE Loss
        mre_loss = torch.abs(y_pred - y_true) / (torch.abs(y_true) + self.epsilon)

        total_loss = self.alpha * log_cosh_loss.mean() + (1 - self.alpha) * mre_loss.mean()
        return total_loss

class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_true, y_pred):
        # compute cosh function
        cosh_loss = torch.cosh(y_pred - y_true)
        log_cosh_loss = torch.log(cosh_loss)
        # return average loss
        return torch.mean(log_cosh_loss)
