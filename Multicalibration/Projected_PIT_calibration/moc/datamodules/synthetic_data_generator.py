import numpy as np
import pandas as pd
import os


class MultivariateRegressionDataset:
    def __init__(self, n_samples=1000, n_features=10, n_targets=3, random_state=None):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_targets = n_targets
        self.rng = np.random.default_rng(random_state)

    def _generate_X(self, dependent=False):
        """
        Generate feature matrix X.

        Parameters:
        ----------
        dependent : bool
            If True, generates dependent features using a random multivariate 
            Gaussian with nontrivial covariance; if False, generates independent 
            standard normal features.

        Description (dependent=True):
        -----------------------------
        1. A random mean vector μ of length p (number of features) is drawn 
        from a standard normal.
        2. A random upper triangular matrix U is generated (entries ~ N(0,1)).
        3. A positive semi-definite covariance matrix Σ = Uᵗ U is computed.
        4. X is sampled from the multivariate normal distribution N(μ, Σ),
        producing dependent feature columns with realistic correlations.
        """
        if dependent:
            mean = self.rng.normal(loc=0.0, scale=5.0, size=self.n_features)
            A = self.rng.normal(size=(self.n_features, self.n_features))
            U = np.triu(A)
            cov = U.T @ U + np.eye(self.n_features) * 1e-6  # stability
            X = self.rng.multivariate_normal(mean=mean, cov=cov, size=self.n_samples)
        else:
            X = self.rng.normal(size=(self.n_samples, self.n_features))
        return X



    def linear_case(self, dependent_X=False):
        """
        Generate a linear multivariate regression dataset.

        Parameters:
        ----------
        dependent_X : bool
            If True, generates the feature matrix X with dependent (correlated) 
            columns using a random multivariate Gaussian.
            If False, generates X with independent standard normal columns.

        Description:
        -----------
        1. Generate X using the _generate_X() method.
        - If dependent_X=True, X has correlated columns.
        - If dependent_X=False, X has independent columns.
        2. Generate a random coefficient matrix B ∈ R^{n_features × n_targets}.
        3. Compute targets Y = X B + ε, where ε is Gaussian noise 
        with small variance added to each target dimension.
        
        Returns:
        -------
        X : ndarray of shape (n_samples, n_features)
            The feature matrix.

        Y : ndarray of shape (n_samples, n_targets)
            The target matrix computed as a linear transformation of X with noise.
        """
        X = self._generate_X(dependent=dependent_X)
        B = self.rng.normal(size=(self.n_features, self.n_targets))
        noise = self.rng.normal(scale=0.1, size=(self.n_samples, self.n_targets))
        Y = X @ B + noise
        return X, Y

    def nonlinear_case(self, dependent_X=False, coupled_outputs=False):
        """
        Generate a nonlinear multivariate regression dataset.

        Parameters:
        ----------
        dependent_X : bool
            If True, generates the feature matrix X with dependent (correlated) columns.
            If False, generates X with independent standard normal columns.

        coupled_outputs : bool
            If True, mixes the target variables after their independent nonlinear generation 
            to introduce coupling (i.e., target dependencies).
            If False, each target is generated independently.

        Description:
        -----------
        1. Generate X using the _generate_X() method.
        2. For each target dimension, compute:
        Y_j = f_j(X β_j) + ε_j,
        where f_j is a randomly chosen nonlinear function 
        (sin, cos, exp, tanh, log1p, inverse quadratic) 
        and β_j is a random coefficient vector.
        3. If coupled_outputs=True, apply an additional random linear mixing between outputs.
        4. Add small Gaussian noise to each target.

        Returns:
        -------
        X : ndarray of shape (n_samples, n_features)
            The feature matrix.

        Y : ndarray of shape (n_samples, n_targets)
            The target matrix computed via nonlinear transformation of X with optional coupling.
        """
        X = self._generate_X(dependent=dependent_X)
        Y = np.zeros((self.n_samples, self.n_targets))
        for j in range(self.n_targets):
            f = self.rng.choice([np.sin, np.cos, np.exp, np.tanh, np.log1p, np.vectorize(lambda x: 1/(1 + x**2))])
            coeffs = self.rng.normal(size=(self.n_features,))
            raw_input = X @ coeffs

            # Safe guards for specific functions
            if f == np.log1p:
                safe_input = np.clip(raw_input, a_min=-0.99, a_max=None)
                nonlinear_part = f(safe_input)
            elif f == np.exp:
                safe_input = np.clip(raw_input, a_min=None, a_max=50)  # prevent overflow
                nonlinear_part = f(safe_input)
            else:
                nonlinear_part = f(raw_input)

            Y[:, j] = nonlinear_part + self.rng.normal(scale=0.1, size=self.n_samples)

        if coupled_outputs:
            Y = Y @ self.rng.normal(size=(self.n_targets, self.n_targets)) + self.rng.normal(scale=0.1, size=(self.n_samples, self.n_targets))

        return X, Y

        

    def latent_variable_case(self, n_latent=2):
        """
        Generate a multivariate regression dataset with shared latent variables.

        Parameters:
        ----------
        n_latent : int
            Number of latent variables that generate both X and Y.

        Description:
        -----------
        1. Generate latent variables Z ~ N(0, I).
        2. Compute:
        X = Z A + ε_x,
        Y = Z C + ε_y,
        where A and C are random coefficient matrices and ε_x, ε_y are Gaussian noise.
        3. This induces shared structure between features and targets, 
        reflecting real-world latent dependency.

        Returns:
        -------
        X : ndarray of shape (n_samples, n_features)
            The feature matrix generated from latent variables.

        Y : ndarray of shape (n_samples, n_targets)
            The target matrix generated from latent variables.
        """

        Z = self.rng.normal(size=(self.n_samples, n_latent))
        A = self.rng.normal(size=(n_latent, self.n_features))
        C = self.rng.normal(size=(n_latent, self.n_targets))
        X = Z @ A + self.rng.normal(scale=0.1, size=(self.n_samples, self.n_features))
        Y = Z @ C + self.rng.normal(scale=0.1, size=(self.n_samples, self.n_targets))
        return X, Y

    def save_dataset(self, X, Y, filename_prefix):
        df_X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        df_Y = pd.DataFrame(Y, columns=[f"y{j}" for j in range(Y.shape[1])])
        df = pd.concat([df_X, df_Y], axis=1)
        df.to_csv(f"{filename_prefix}.csv", index=False)
        # np.savez(f"{filename_prefix}.npz", X=X, Y=Y)
        # print(f"Saved {filename_prefix}.csv and {filename_prefix}.npz")
        print(f"Saved {filename_prefix}.csv")

    def generate_multiple_datasets(self, n_files, n_samples_range, n_features_range, 
                                n_targets_range, destination, mode='linear', 
                                dependent_X=False, coupled_outputs=False):
        """
        Generate and save multiple synthetic datasets with varying sizes.

        Parameters:
        ----------
        n_files : int
            Number of datasets to generate.

        n_samples_range : tuple of (int, int)
            Range of number of samples (min, max).

        n_features_range : tuple of (int, int)
            Range of number of features (min, max).

        n_targets_range : tuple of (int, int)
            Range of number of targets (min, max).

        destination : str
            Directory path where the datasets will be saved.

        mode : str
            Type of dataset: 'linear', 'nonlinear', or 'latent'.

        dependent_X : bool
            Whether to generate dependent features (for linear and nonlinear).

        coupled_outputs : bool
            Whether to couple outputs (only for nonlinear).
        """
        os.makedirs(destination, exist_ok=True)

        for i in range(n_files):
            n_samples = self.rng.integers(*n_samples_range)
            n_features = self.rng.integers(*n_features_range)
            n_targets = self.rng.integers(*n_targets_range)

            # Update object’s dimensions
            self.n_samples = n_samples
            self.n_features = n_features
            self.n_targets = n_targets

            # Generate data
            if mode == 'linear':
                X, Y = self.linear_case(dependent_X=dependent_X)
            elif mode == 'nonlinear':
                X, Y = self.nonlinear_case(dependent_X=dependent_X, coupled_outputs=coupled_outputs)
            elif mode == 'latent':
                X, Y = self.latent_variable_case()
            else:
                raise ValueError("Invalid mode. Choose 'linear', 'nonlinear', or 'latent'.")

            # Save to CSV
            filename_prefix = os.path.join(destination, f"{mode}_dataset_{i}")
            filename_prefix = os.path.join(destination,
                                           f"{mode}_depX-{dependent_X}_coup-{coupled_outputs}_nS-{n_samples}_nF-{n_features}_nT-{n_targets}_id-{i}")

            self.save_dataset(X, Y, filename_prefix)

if __name__ == "__main__":
    # Initialize generator
    gen = MultivariateRegressionDataset(random_state=42)

    # Common ranges
    n_files = 10
    n_samples_range = (500, 5000)
    n_features_range = (10, 50)
    n_targets_range = (2, 10)
    destination_base = "generated_datasets"

    os.makedirs(destination_base, exist_ok=True)

    configs = [
        ("linear", False, False),  # Linear, independent X
        # ("linear", True, False),   # Linear, dependent X
        # ("nonlinear", False, False),  # Nonlinear, independent X, uncoupled outputs
        # ("nonlinear", False, True),   # Nonlinear, independent X, coupled outputs
        # ("nonlinear", True, False),   # Nonlinear, dependent X, uncoupled outputs
        # ("nonlinear", True, True),    # Nonlinear, dependent X, coupled outputs
        # ("latent", False, False),     # Latent (X dependency not applicable)
    ]

    for mode, dependent_X, coupled_outputs in configs:
        destination = os.path.join(destination_base, mode)
        os.makedirs(destination, exist_ok=True)

        for i in range(n_files):
            n_samples = gen.rng.integers(*n_samples_range)
            n_features = gen.rng.integers(*n_features_range)
            n_targets = gen.rng.integers(*n_targets_range)

            # Update generator dimensions
            gen.n_samples = n_samples
            gen.n_features = n_features
            gen.n_targets = n_targets

            # Generate data
            if mode == 'linear':
                X, Y = gen.linear_case(dependent_X=dependent_X)
            elif mode == 'nonlinear':
                X, Y = gen.nonlinear_case(dependent_X=dependent_X, coupled_outputs=coupled_outputs)
            elif mode == 'latent':
                X, Y = gen.latent_variable_case()
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            # Build descriptive filename
            filename_prefix = os.path.join(
                destination,
                f"{mode}_depX-{dependent_X}_coup-{coupled_outputs}_nS-{n_samples}_nF-{n_features}_nT-{n_targets}_id-{i}"
            )

            # Save dataset
            gen.save_dataset(X, Y, filename_prefix)
