import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize

class BrierScoreLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        """
        Compute the Brier Score loss.

        :param y_pred: Tensor of shape (batch_size, num_classes), predicted raw logits
        :param y_true: Tensor of shape (batch_size,), true class labels (not one-hot yet)
        :return: Scalar loss value
        """
        num_classes = y_pred.size(1)

        # Convert y_true to one-hot encoding
        y_true_one_hot = F.one_hot(y_true, num_classes=num_classes).float()
        y_pred = F.softmax(y_pred, dim=1)
        # Compute squared difference
        loss = torch.mean(torch.sum((y_pred - y_true_one_hot) ** 2, dim=1))
        return loss
    
class RPS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        """
        Compute the Ranked Probability Score.

        :param y_pred: Tensor of shape (batch_size, num_classes), predicted raw logits
        :param y_true: Tensor of shape (batch_size,), true class labels (not one-hot yet)
        :return: Scalar loss value
        """
        num_classes = y_pred.size(1)
        
        # Convert y_true to one-hot encoding
        y_true_one_hot = F.one_hot(y_true, num_classes=num_classes).float()
        y_true_cdf = torch.cumsum(y_true_one_hot, dim=1)
        # Convert y_pred to probabilities
        y_pred = F.softmax(y_pred, dim=1)
        y_pred_cdf = torch.cumsum(y_pred, dim=1)

        # Compute squared difference
        score = torch.mean(torch.sum((y_pred_cdf - y_true_cdf) ** 2, dim=1)/(num_classes-1))
        return score
    
class TemperatureScaler(nn.Module):
    #takes logits as inputs and scales them by a learnable parameter
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)  # Initialize T=1.0

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits, labels, criterion, max_iter = 50):
        """
        Optimize the temperature parameter to minimize the loss (scoring rule) on the validation set.
        """
        self.temperature.requires_grad = True  # Ensure T is learnable
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def eval():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(eval)  # Run L-BFGS optimization

    def transform(self, logits):
        return logits / self.temperature