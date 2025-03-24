import numpy as np

# Classe pour le modèle Gaussien
class GaussianModel:
    def __init__(self, mean, covariance, n_samples):
        self.mean = mean
        self.covariance = covariance
        self.n_samples = n_samples
        self.name = "Gaussian Model"

    def generate_samples(self):
        return np.random.multivariate_normal(self.mean, self.covariance, self.n_samples)


# Classe pour le modèle uniforme multivarié
class UniformModel:
    def __init__(self, n_samples, lower_bounds, upper_bounds):
        """
        Initialise le modèle uniforme multivarié avec des bornes spécifiques pour chaque composante.

        - n_samples: Le nombre d'échantillons à générer
        - lower_bounds: Un vecteur de taille d représentant les bornes inférieures pour chaque composante
        - upper_bounds: Un vecteur de taille d représentant les bornes supérieures pour chaque composante
        """
        self.n_samples = n_samples
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.name = "Uniform Model"

    def generate_samples(self):
        """
        Génère des échantillons à partir de distributions uniformes indépendantes, avec des bornes spécifiques.
        """
        # Générer des échantillons indépendants pour chaque composante, selon les bornes définies
        samples = np.random.uniform(low=self.lower_bounds, high=self.upper_bounds, size=(self.n_samples, len(self.lower_bounds)))
        return samples

import numpy as np

# Class for the Gaussian model
class GaussianModel:
    def __init__(self, mean, covariance, n_samples):
        self.mean = mean
        self.covariance = covariance
        self.n_samples = n_samples
        self.name = "Gaussian Model"

    def generate_samples(self):
        return np.random.multivariate_normal(self.mean, self.covariance, self.n_samples)


# Class for the multivariate uniform model
class UniformModel:
    def __init__(self, n_samples, lower_bounds, upper_bounds):
        """
        Initializes the multivariate uniform model with specific bounds for each component.

        - n_samples: The number of samples to generate
        - lower_bounds: A vector of size d representing the lower bounds for each component
        - upper_bounds: A vector of size d representing the upper bounds for each component
        """
        self.n_samples = n_samples
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.name = "Uniform Model"

    def generate_samples(self):
        """
        Generates samples from independent uniform distributions with specific bounds.
        """
        # Generate independent samples for each component, according to the defined bounds
        samples = np.random.uniform(low=self.lower_bounds, high=self.upper_bounds, size=(self.n_samples, len(self.lower_bounds)))
        return samples

