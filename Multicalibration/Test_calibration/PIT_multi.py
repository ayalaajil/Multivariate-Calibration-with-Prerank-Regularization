import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.decomposition import PCA


# Function to project the samples onto the vectors
def proj_for(x_values, u, sample):
    sample_proj = sample.dot(u)
    
    if x_values is None:
        x_values = sample

    x_proj = x_values.dot(u)
    sample_sorted = np.sort(sample_proj)
    n = len(sample)
    
    cdf_values = np.searchsorted(sample_sorted, x_proj, side='right') / n
    return x_proj, cdf_values

# Function to calculate PIT
def calculate_pit(values, u, sample):
    return proj_for(values, u, sample)[1]

# Function to display PIT plots
def plot_pit(pit_values, dimension, pca, n_samples, name):
    cols = 3  # Fixes 3 plots per row
    rows = int(np.ceil(dimension / cols))  # Number of rows needed

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))  # Dynamically adjust the size

    # Ensure `axs` is always a 2D array to avoid errors
    axs = np.array(axs).reshape(rows, cols)

    for i in range(dimension):
        row, col = divmod(i, cols)  # Find the position in the grid
        ax = axs[row, col]  # Get the corresponding axis
        
        ax.hist(pit_values[i], bins=10, density=True, alpha=0.6, color='g')
        title = f"PIT - Direction according to component ${i+1}$" if pca else f"PIT - Direction according to $e_{i+1}$"
        ax.set_title(title)
        ax.set_xlabel("Projected PIT Value")
        ax.set_ylabel("Density")

    # Remove empty subplots if `dimension` is not a multiple of 3
    for i in range(dimension, rows * cols):
        fig.delaxes(axs.flatten()[i])

    plt.tight_layout()  # Automatically adjust the display
    filename = f"PIT_proj_PCA_{n_samples}_{name}.png" if pca else f"PIT_proj_{n_samples}_{name}.png"
    plt.savefig(filename)
    plt.show()


def create_exp_correlation_matrix(d, diagonal_value=1.0, off_diagonal_value=np.exp(-1)):
    return np.array([[diagonal_value if i == j else off_diagonal_value for j in range(d)] for i in range(d)])

def create_mean_vector(d, value=0):
    return np.full(d, value)


# Flags and parameters
pca_flag = 1
normal_flag = 1
d = 3
n_samples = 5000

mean = create_mean_vector(d)
mean_hat = create_mean_vector(d, value=0.5)
covariance = create_exp_correlation_matrix(d)
covariance_hat = create_exp_correlation_matrix(d)

model = GaussianModel(mean, covariance, n_samples)
'''lower_bounds = np.array([0, 1, -2])  # Lower bounds for each component
upper_bounds = np.array([1, 3, 2])   # Upper bounds for each component

# Create a uniform model
model = UniformModel(n_samples, lower_bounds, upper_bounds)'''
Y = model.generate_samples()

# Choose the model and generate the samples
if normal_flag:
    model_hat = GaussianModel(mean_hat, covariance_hat, n_samples)
    Y_hat = model_hat.generate_samples()
    name = "RANK_Normal_var_overestimated"
else:
    model_hat = NormalModel(n_samples)
    Y_hat = model_hat.generate_samples()
    name = "Non-normal"

# PCA ou identity
if pca_flag:
    pca = PCA(n_components=d)
    pca.fit(Y_hat)
    vectors = pca.components_
else:
    vectors = np.eye(d)

# Calculating the PIT
pit_values = [calculate_pit(Y, vectors[i], Y_hat) for i in range(d)]

# Plotting the PIT
plot_pit(pit_values, d, pca_flag, n_samples, name)
