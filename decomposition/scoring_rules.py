import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from typing import Callable, Optional, List

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
        score = torch.mean(torch.sum((y_pred_cdf - y_true_cdf) ** 2, dim=1))
        return score
    
class KLDivergence(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, calibrated_logits):
        """
        Compute the Kullback-Leibler Divergence.

        :param logits: Tensor of shape (batch_size, num_classes), predicted raw logits
        :param calibrated_logits: Tensor of shape (batch_size,), calibrated logits
        :return: Scalar loss value
        """
        log_probs = F.log_softmax(logits, dim=-1)

        # Convert targets (calibrated logits) to probabilities
        target_probs = F.softmax(calibrated_logits, dim=-1)

        # Compute KL divergence loss
        loss = F.kl_div(log_probs, target_probs, reduction='batchmean')

        return loss
    
class MSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, scaled_logits):
        """
        Compute the Mean Squared Error.

        :param logits: Tensor of shape (batch_size, num_classes), predicted raw logits
        :param scaled_logits: Tensor of shape (batch_size,), calibrated logits
        :return: Scalar loss value
        """
        y_pred = F.softmax(logits, dim=1)
        y_true = F.softmax(scaled_logits, dim=1)
        loss = F.mse_loss(y_pred, y_true)
        return loss
    
class RPSDivergence(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, scaled_logits):
        """
        Compute the Ranked Probability Score Divergence.

        :param logits: Tensor of shape (batch_size, num_classes), predicted raw logits
        :param scaled_logits: Tensor of shape (batch_size,), calibrated logits
        :return: Scalar loss value
        """
        
        # Convert y_true to one-hot encoding
        scaled_probs = F.softmax(scaled_logits, dim=1)
        scaled_probs_cdf = torch.cumsum(scaled_probs, dim=1)
        # Convert y_pred to probabilities
        probs = F.softmax(logits, dim=1)
        probs_cdf = torch.cumsum(probs, dim=1)

        # Compute squared difference
        loss = torch.mean(torch.sum((probs_cdf - scaled_probs_cdf) ** 2, dim=1))
        return loss
    
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


class TemperatureScalerBisection(nn.Module):
    def __init__(self, max_bisection_steps=30):
        """
        Implements Temperature Scaling with Bisection Search.

        Args:
            init_temp (float): Initial temperature scaling value.
            bisection_range (tuple): (lower, upper) range for the log-temperature search.
            max_bisection_steps (int): Number of steps for bisection search.
        """
        super().__init__()
        self.max_bisection_steps = max_bisection_steps

    def _get_loss_grad(self, invtemp, logits, labels, criterion):
        logits = logits.clone().detach().requires_grad_(True)  # Ensure logits require grad
    
        with torch.enable_grad():  # Ensure PyTorch autograd is tracking computations
            invtemp_tensor = torch.tensor(invtemp, dtype=logits.dtype, device=logits.device, requires_grad=True)

            # Apply inverse temperature scaling
            scaled_logits = logits * invtemp_tensor

            # Compute loss
            loss = criterion(scaled_logits, labels)

            # Compute gradient w.r.t invtemp
            loss.backward()

            # Extract gradient value
            grad = invtemp_tensor.grad.item()

        return grad

    def _bisection_search(self, f: Callable[[float], float], a: float, b: float, n_steps: int):
        """
        Performs bisection search to find the optimal inverse temperature.

        Args:
            logits (Tensor): Logits from the model (before softmax).
            labels (Tensor): True class labels.
            criterion (callable): Loss function to be minimized.
        
        Returns:
            float: Optimal temperature scaling factor.
        """
        for _ in range(n_steps):
            c = a + 0.5 * (b - a)
            f_c = f(c)
            if f_c > 0:
                b = c
            else:
                a = c

        return 0.5 * (a + b)

    def fit(self, logits, labels, criterion):
        """
        Fits the temperature parameter using bisection search.

        Args:
            logits (Tensor): Logits from the model (before softmax).
            labels (Tensor): True class labels.
            criterion (callable): Loss function to be minimized.
        """
        objective_grad = lambda u, l=logits, tar=labels: self._get_loss_grad(np.exp(u), l, tar, criterion)

        # should reach about float32 accuracy
        # need log_2(32) = 5 steps to get to length 1 and then 24 more steps to get to float32 epsilon (2^{-24})
        self.invtemp_ = np.exp(self._bisection_search(objective_grad, a=-16, b=16, n_steps=self.max_bisection_steps))
        # print(f'{self.invtemp_=}')

    def transform(self, logits):
        """
        Applies the learned temperature scaling to logits.

        Args:
            logits (Tensor): Logits from the model (before softmax).
        
        Returns:
            Tensor: Scaled logits.
        """
        return self.invtemp_ * logits

    def predict_proba(self, logits):
        """
        Returns calibrated probabilities after applying temperature scaling.

        Args:
            logits (Tensor): Logits from the model (before softmax).
        
        Returns:
            Tensor: Softmax probabilities.
        """
        return torch.softmax(self.forward(logits), dim=-1)
