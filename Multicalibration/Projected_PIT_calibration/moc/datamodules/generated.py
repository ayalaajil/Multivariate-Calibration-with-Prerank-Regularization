import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from .base_datamodule import BaseDataModule
from .synthetic_data_generator import MultivariateRegressionDataset

class GeneratedDataModule(BaseDataModule):
    def get_data(self):
        generator = MultivariateRegressionDataset(
            n_samples=self.rc.config.synthetic.n_samples,
            n_features=self.rc.config.synthetic.n_features,
            n_targets=self.rc.config.synthetic.n_targets,
            random_state=self.rc.seed
        )
        # Choose one generation mode
        if self.rc.config.synthetic.mode == "linear":
            X, Y = generator.linear_case(dependent_X=self.rc.config.synthetic.dependent_X)
        elif self.rc.config.synthetic.mode == "nonlinear":
            X, Y = generator.nonlinear_case(dependent_X=self.rc.config.synthetic.dependent_X,
                                            coupled_outputs=self.rc.config.synthetic.coupled_outputs)
        elif self.rc.config.synthetic.mode == "latent":
            X, Y = generator.latent_variable_case(n_latent=self.rc.config.synthetic.n_latent)
        else:
            raise ValueError(f"Unknown synthetic mode: {self.rc.config.synthetic.mode}")
        return X, Y
    # def __init__(self, csv_path, batch_size=256, test_size=0.2, val_size=0.1, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.csv_path = csv_path
    #     self.batch_size = batch_size
    #     self.test_size = test_size
    #     self.val_size = val_size

    # def get_data(self, seed=0):
    #     df = pd.read_csv(self.csv_path)
    #     X = df[[col for col in df.columns if col.startswith("x")]].values.astype("float32")
    #     Y = df[[col for col in df.columns if col.startswith("y")]].values.astype("float32")

    #     X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=self.test_size + self.val_size, random_state=seed)
    #     val_frac = self.val_size / (self.test_size + self.val_size)
    #     X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=val_frac, random_state=seed)

    #     self.input_dim = X.shape[1]
    #     self.output_dim = Y.shape[1]

    #     self.train_data = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    #     self.val_data = TensorDataset(torch.tensor(X_val), torch.tensor(Y_val))
    #     self.test_data = TensorDataset(torch.tensor(X_test), torch.tensor(Y_test))

    # def train_dataloader(self):
    #     return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    # def val_dataloader(self):
    #     return DataLoader(self.val_data, batch_size=self.batch_size)

    # def test_dataloader(self):
    #     return DataLoader(self.test_data, batch_size=self.batch_size)
