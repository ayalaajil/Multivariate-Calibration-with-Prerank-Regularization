import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data import TensorDataset, DataLoader
from scoring_rules import BrierScoreLoss, RPS, TemperatureScaler, KLDivergence, MSE, RPSDivergence, TemperatureScalerBisection
from probmetrics.calibrators import get_calibrator
from probmetrics.distributions import CategoricalLogits
import wandb

from preprocessing import get_realmlp_td_s_pipeline


class ScalingLayer(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale[None, :]


class NTPLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, zero_init: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factor = 0.0 if zero_init else 1.0
        self.weight = nn.Parameter(factor * torch.randn(in_features, out_features))
        self.bias = nn.Parameter(factor * torch.randn(1, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (1. / np.sqrt(self.in_features)) * (x @ self.weight) + self.bias


class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul(torch.tanh(torch.nn.functional.softplus(x)))


class SimpleMLP(BaseEstimator):
    def __init__(self, is_classification: bool, device: str = 'cpu', loss_type = 'cross_entropy', ts_type = "lbfgs"):
        self.is_classification = is_classification
        self.device = device
        self.loss_type = loss_type
        self.ts_type = ts_type

    def fit(self, X, y, X_val=None, y_val=None, X_test=None, y_test=None):

        # wandb.init(project="scoring-rules-realMLP", config={"epochs": 150, "learning_rate": 0.001 if self.is_classification else 0.07})

        input_dim = X.shape[1]
        is_classification = self.is_classification

        output_dim = 1 if len(y.shape) == 1 else y.shape[1]

        if self.is_classification:
            self.class_enc_ = OrdinalEncoder(dtype=np.int64)
            y = self.class_enc_.fit_transform(y[:, None])[:, 0]
            self.classes_ = self.class_enc_.categories_[0]
            output_dim = len(self.class_enc_.categories_[0])
            if y_val is not None:
                y_val = self.class_enc_.transform(y_val[:, None])[:, 0]
            if y_test is not None:
                y_test = self.class_enc_.transform(y_test[:, None])[:, 0]
        else:  # standardize targets
            self.y_mean_ = np.mean(y, axis=0)
            self.y_std_ = np.std(y, axis=0)
            y = (y - self.y_mean_) / (self.y_std_ + 1e-30)
            if y_val is not None:
                y_val = (y_val - self.y_mean_) / (self.y_std_ + 1e-30)
            if y_test is not None:
                y_test = (y_test - self.y_mean_) / (self.y_std_ + 1e-30)

        act = nn.SELU if is_classification else Mish
        model = nn.Sequential(
            ScalingLayer(input_dim),
            NTPLinear(input_dim, 256), act(),
            NTPLinear(256, 256), act(),
            NTPLinear(256, 256), act(),
            NTPLinear(256, output_dim, zero_init=True),
        ).to(self.device)

        if self.loss_type == 'cross_entropy':
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1) if is_classification else nn.MSELoss()
            divergence = KLDivergence()
        elif self.loss_type == 'brier':
            criterion = BrierScoreLoss()
            divergence = MSE()
        elif self.loss_type == 'rps':
            criterion = RPS()
            divergence = RPSDivergence()
        else:
            raise ValueError(f"Unknown loss type: {self.loss}")
        
        if self.ts_type=="lbfgs":
            temp_scaler = TemperatureScaler().to(self.device)
        elif self.ts_type=="bisection":
            temp_scaler = TemperatureScalerBisection().to(self.device)
        else:
            raise ValueError(f"Unknown TS type: {self.ts_type}")
        
        params = list(model.parameters())
        scale_params = [params[0]]
        weights = params[1::2]
        biases = params[2::2]
        opt = torch.optim.Adam([dict(params=scale_params), dict(params=weights), dict(params=biases)],
                                betas=(0.9, 0.95))

        x_train = torch.as_tensor(X, dtype=torch.float32)
        y_train = torch.as_tensor(y, dtype=torch.int64 if self.is_classification else torch.float32)
        if not is_classification and len(y_train.shape) == 1:
            y_train = y_train[:, None]

        if X_val is not None and y_val is not None:
            x_valid = torch.as_tensor(X_val, dtype=torch.float32)
            y_valid = torch.as_tensor(y_val, dtype=torch.int64 if self.is_classification else torch.float32)
            if not is_classification and len(y_valid.shape) == 1:
                y_valid = y_valid[:, None]
        else:
            x_valid = x_train[:0] #retain the number of columns but the number of rows is 0
            y_valid = y_train[:0]

        if X_test is not None and y_test is not None:
            x_test = torch.as_tensor(X_test, dtype=torch.float32)
            y_test = torch.as_tensor(y_test, dtype=torch.int64 if self.is_classification else torch.float32)
            # if not is_classification and len(y_test.shape) == 1:
            #     y_test = y_test[:, None]
        else:    
            x_test = x_train[:0]
            y_test = y_train[:0]

        train_ds = TensorDataset(x_train, y_train)
        valid_ds = TensorDataset(x_valid, y_valid)
        test_ds = TensorDataset(x_test, y_test)

        n_train = x_train.shape[0]
        n_valid = x_valid.shape[0]
        n_test = x_test.shape[0]
        n_epochs = 150
        train_batch_size = min(256, n_train)
        valid_batch_size = max(1, min(1024, n_valid))
        test_batch_size = max(1, min(1024, n_test))
        
        train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, drop_last=True)
        valid_dl = DataLoader(valid_ds, batch_size=valid_batch_size, shuffle=False)
        test_dl = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False)
    
        n_train_batches = len(train_dl)
        base_lr = 0.001 if is_classification else 0.07
        best_valid_loss = np.Inf
        best_valid_params = None
        train_losses = []
        test_losses = []
        refinement_losses = []
        calib_losses = []

        for epoch in range(n_epochs):
            # print(f'Epoch {epoch + 1}/{n_epochs}')
            model.train()
            train_loss = 0.0
            for batch_idx, (x_batch, y_batch) in enumerate(train_dl):
                # set learning rates according to schedule
                t = (epoch * n_train_batches + batch_idx) / (n_epochs * n_train_batches)
                lr_sched_value = 0.5 - 0.5 * np.cos(2 * np.pi * np.log2(1 + 15 * t))
                lr = base_lr * lr_sched_value
                # print(f'{lr=:g}')
                opt.param_groups[0]['lr'] = 6 * lr  # for scale
                opt.param_groups[1]['lr'] = lr  # for weights
                opt.param_groups[2]['lr'] = 0.1 * lr  # for biases

                # optimization
                y_pred = model(x_batch.to(self.device))
                loss = criterion(y_pred, y_batch.to(self.device))
                loss.backward()
                opt.step()
                opt.zero_grad()
                train_loss += loss.item()
            avg_train_loss = train_loss/n_train_batches

            # save parameters if validation score improves
            model.eval()
            logits_list, labels_list = [], []
            #I use validation set just to learn \beta parameter from TS
            with torch.no_grad():
                for x_batch, y_batch in valid_dl:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    y_pred = model(x_batch)  # Get raw logits
                    logits_list.append(y_pred)
                    labels_list.append(y_batch)
                logits = torch.cat(logits_list).to(self.device)
                labels = torch.cat(labels_list).to(self.device)

                valid_loss = criterion(logits, labels).item()
                #Using probmetrics library to do TS, comment if you want to do L-BFGS optimization
                # calib = get_calibrator('temp-scaling')
                # calib.fit_torch(CategoricalLogits(logits), labels)
                #Uncomment these two lines if you want to do L-BFGS optimization
                temp_scaler.fit(logits, labels, criterion)
                
                #Calculating the losses on test set
                test_loss, refinement_loss, calib_loss = 0.0, 0.0, 0.0
                for x_batch, y_batch in test_dl:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    y_pred = model(x_batch) #raw logits
                    scaled_ypred = temp_scaler.transform(y_pred)
                    # scaled_ypred = calib.predict_proba_torch(CategoricalLogits(y_pred)).get_logits()

                    loss = criterion(y_pred, y_batch)
                    ref_loss = criterion(scaled_ypred, y_batch)
                    cal_loss = divergence(y_pred, scaled_ypred)

                    test_loss += loss.item()
                    refinement_loss += ref_loss.item()
                    calib_loss += cal_loss.item()
            
                avg_test_loss = test_loss/len(test_dl)
                avg_refinement_loss = refinement_loss/len(test_dl)
                avg_calib_loss = calib_loss/len(test_dl)
                
                train_losses.append(avg_train_loss)
                test_losses.append(avg_test_loss)
                refinement_losses.append(avg_refinement_loss)
                calib_losses.append(avg_calib_loss)
                # refinement_loss = criterion(scaled_logits, labels).item()
                # calib_loss = valid_loss - refinement_loss
                
                # wandb.log({"train_loss": avg_train_loss, "test_loss": avg_test_loss, "valid_loss": valid_loss,
                #            "refinement_error" : avg_refinement_loss, "calibration_error": calib_loss, "epoch": epoch + 1})
                
                if valid_loss <= best_valid_loss:  # use <= for last best epoch
                    best_valid_loss = valid_loss
                    best_valid_params = [p.detach().clone() for p in model.parameters()]

        # after training, revert to best epoch
        with torch.no_grad():
            for p_model, p_copy in zip(model.parameters(), best_valid_params):
                p_model.set_(p_copy)

        self.model_ = model
        # wandb.finish()
        print("Returning from fit() method")
        return self, train_losses, test_losses, refinement_losses, calib_losses

    def predict(self, X):
        x = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        self.model_.eval()
        with torch.no_grad():
            y_pred = self.model_(x).cpu().numpy()
        if self.is_classification:
            # return classes with highest probability
            return self.class_enc_.inverse_transform(np.argmax(y_pred, axis=-1)[:, None])[:, 0]
        else:
            return y_pred[:, 0] * self.y_std_ + self.y_mean_

    def predict_proba(self, X):
        assert self.is_classification
        self.model_.eval()
        x = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            y_pred = torch.softmax(self.model_(x), dim=-1).cpu().numpy()
        return y_pred


class Standalone_RealMLP_TD_S_Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, device: str = 'cpu', loss_type = 'cross_entropy', ts_type = "lbfgs"):
        self.device = device
        self.loss_type = loss_type
        self.ts_type = ts_type

    def fit(self, X, y, X_val=None, y_val=None, X_test=None, y_test=None):
        self.prep_ = get_realmlp_td_s_pipeline()
        self.model_ = SimpleMLP(is_classification=True, device=self.device, loss_type=self.loss_type, ts_type=self.ts_type)
        X = self.prep_.fit_transform(X)
        if X_val is not None:
            X_val = self.prep_.transform(X_val)
        if X_test is not None:
            X_test = self.prep_.transform(X_test)
        model, train_loss, test_loss, refinement_loss, calib_loss = self.model_.fit(X, y, X_val, y_val, X_test, y_test)
        self.classes_ = self.model_.classes_
        return model, train_loss, test_loss, refinement_loss, calib_loss

    def predict(self, X):
        return self.model_.predict(self.prep_.transform(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(self.prep_.transform(X))


class Standalone_RealMLP_TD_S_Regressor(BaseEstimator, RegressorMixin):
    def __init__(self, device: str = 'cpu'):
        self.device = device

    def fit(self, X, y, X_val=None, y_val=None):
        self.prep_ = get_realmlp_td_s_pipeline()
        self.model_ = SimpleMLP(is_classification=False, device=self.device)
        X = self.prep_.fit_transform(X)
        if X_val is not None:
            X_val = self.prep_.transform(X_val)
        self.model_.fit(X, y, X_val, y_val)

    def predict(self, X):
        return self.model_.predict(self.prep_.transform(X))
