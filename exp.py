# Import packages
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import chi2

# Set seed for reproducibility
SEED = 19
np.random.seed(SEED)
torch.manual_seed(SEED)

# Data: Longleaf pine (from spatstat R package)
# 584 locations and diameters at breast height (DBH) of Longleaf pine trees
# in a 200 x 200 metre region in southern Georgia (USA).
# Fields: x, y = locations; marks = DBH.
#
# IPPP assumptions used here:
# - λ(s) ≥ 0
# - counts in disjoint regions are independent
# - expected count in W is ∫_W λ(u) du
df = pd.read_csv("data/longleaf.csv")
coords_np = df[["x", "y"]].values.astype(np.float32)

# Observation window
x_min, x_max = coords_np[:, 0].min(), coords_np[:, 0].max()
y_min, y_max = coords_np[:, 1].min(), coords_np[:, 1].max()
area = (x_max - x_min) * (y_max - y_min)
n = len(coords_np)

# Normalize coordinates consistently for model inputs
mean = coords_np.mean(axis=0)
std = coords_np.std(axis=0)

# ---------------------------------------------------
# Berman-Turner quadrature grid
# ---------------------------------------------------
# The IPPP likelihood is approximated by aggregating the point pattern
# over grid cells. Each cell contributes:
#
#   Y_j ~ Poisson(w_j λ(u_j))
#
# where:
#   Y_j = number of observed points in cell j
#   u_j = cell center
#   w_j = cell area
#
# This is equivalent to a weighted Poisson regression approximation
# to the IPPP likelihood.
def make_berman_turner_grid(n_per_dim=100):
    x_edges = np.linspace(x_min, x_max, n_per_dim + 1)
    y_edges = np.linspace(y_min, y_max, n_per_dim + 1)

    # Cell counts: Y_j
    counts, _, _ = np.histogram2d(
        coords_np[:, 0],
        coords_np[:, 1],
        bins=[x_edges, y_edges]
    )

    # Cell centers: u_j
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    gx, gy = np.meshgrid(x_centers, y_centers, indexing="ij")
    grid_raw = np.column_stack([gx.ravel(), gy.ravel()])

    # Normalize grid cell centers using the same transform as the data
    grid_norm = (grid_raw - mean) / std

    y_counts = counts.ravel().astype(np.float32)

    cell_area = area / (n_per_dim * n_per_dim)
    weights = np.full_like(y_counts, cell_area, dtype=np.float32)

    return (
        torch.tensor(grid_norm, dtype=torch.float32),
        torch.tensor(y_counts[:, None], dtype=torch.float32),
        torch.tensor(weights[:, None], dtype=torch.float32)
    )


quad_coords, quad_counts, quad_weights = make_berman_turner_grid(n_per_dim=100)

# ---------------------------------------------------
# Models
# ---------------------------------------------------

class ConstantIPPP(nn.Module):
    """
    Constant-intensity IPPP.

    This is the neural equivalent of the homogeneous Poisson point process:

        λ(s) = λ0

    It has one parameter and should match the closed-form HPPP MLE when
    initialized at n / |W|.
    """
    def __init__(self, lambda_init):
        super().__init__()
        self.log_lambda = nn.Parameter(
            torch.tensor(np.log(lambda_init), dtype=torch.float32)
        )

    def forward(self, x):
        lambda0 = torch.exp(self.log_lambda)
        return lambda0.expand(x.shape[0], 1)


class LinearIPPP(nn.Module):
    """
    Log-linear inhomogeneous Poisson point process.

    This model allows first-order spatial variation in intensity:

        λ(s) = exp(β0 + β1 x + β2 y)

    The model is initialized at the HPPP solution by setting β1 = β2 = 0
    and β0 = log(lambda_hat). Training then estimates a linear spatial trend.
    """
    def __init__(self, input_dim, lambda_init):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.fill_(np.log(lambda_init))

    def forward(self, x):
        eta = self.linear(x)
        eta = torch.clamp(eta, min=-20, max=20)
        return torch.exp(eta)

# ---------------------------------------------------
# Berman-Turner likelihood
# ---------------------------------------------------
# Constants that do not depend on model parameters are omitted.
#
# Approximate log-likelihood:
#
#   Σ_j [Y_j log λ(u_j) - w_j λ(u_j)]
#
# where Y_j is the cell count and w_j is the cell area.
def bt_loglik(model):
    lambda_vals = model(quad_coords)
    return (
        quad_counts * torch.log(lambda_vals + 1e-8)
        - quad_weights * lambda_vals
    ).sum().item()


def bt_nll(model):
    lambda_vals = model(quad_coords)
    return -(
        quad_counts * torch.log(lambda_vals + 1e-8)
        - quad_weights * lambda_vals
    ).sum()

# ---------------------------------------------------
# Fit helper
# ---------------------------------------------------
def fit(model, epochs=2000, lr=1e-3):
    opt = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        opt.zero_grad()
        loss = bt_nll(model)
        loss.backward()
        opt.step()

    return model

# ---------------------------------------------------
# Fit models
# ---------------------------------------------------

lambda_hat = n / area

# Model 1: HPPP closed-form baseline
#
# Homogeneous Poisson point process:
#
#   λ(s) = λ_hat = n / |W|
#
# This is the maximum likelihood estimate under constant intensity.
loglik_hppp = n * np.log(lambda_hat) - lambda_hat * area

# Model 2: Constant NN
#
# Same model class as HPPP, but estimated through PyTorch optimization.
# This is a sanity check: it should recover the HPPP log-likelihood.
const_model = fit(
    ConstantIPPP(lambda_hat),
    epochs=1000,
    lr=1e-3
)
loglik_const = bt_loglik(const_model)

# Model 3: Linear IPPP
#
# Log-linear spatial intensity model:
#
#   λ(s) = exp(β0 + β1 x + β2 y)
#
# This tests whether a first-order spatial trend improves fit over
# homogeneous intensity.
lin_model = fit(
    LinearIPPP(input_dim=2, lambda_init=lambda_hat),
    epochs=3000,
    lr=1e-3
)
loglik_lin = bt_loglik(lin_model)

# ---------------------------------------------------
# Model comparison
# ---------------------------------------------------
def aic(ll, k):
    return 2 * k - 2 * ll

def bic(ll, k, n):
    return k * np.log(n) - 2 * ll

def lr_stat(ll_full, ll_reduced):
    return 2 * (ll_full - ll_reduced)

k_hppp = 1
k_const = 1
k_lin = 3

results_df = pd.DataFrame({
    "Model": ["HPPP", "Constant NN", "Linear IPPP"],
    "k": [k_hppp, k_const, k_lin],
    "BT_LogLik": [loglik_hppp, loglik_const, loglik_lin],
    "AIC": [
        aic(loglik_hppp, k_hppp),
        aic(loglik_const, k_const),
        aic(loglik_lin, k_lin)
    ],
    "BIC": [
        bic(loglik_hppp, k_hppp, n),
        bic(loglik_const, k_const, n),
        bic(loglik_lin, k_lin, n)
    ]
})

print(results_df)

# ---------------------------------------------------
# Likelihood ratio test
# ---------------------------------------------------
# Linear IPPP has two additional parameters relative to HPPP:
# β1 and β2. Therefore df = 2.
lr_lin_vs_hppp = lr_stat(loglik_lin, loglik_hppp)
p_lin_vs_hppp = chi2.sf(lr_lin_vs_hppp, df=2)

print("\nLikelihood Ratio Test:")
print(f"Linear vs HPPP LR stat: {lr_lin_vs_hppp:.4f}")
print(f"Linear vs HPPP p-value: {p_lin_vs_hppp:.6g}")

print("\nSanity checks:")
print(f"lambda_hat: {lambda_hat:.6f}")
print(f"HPPP loglik: {loglik_hppp:.4f}")
print(f"Constant NN loglik: {loglik_const:.4f}")
print(f"Linear IPPP loglik: {loglik_lin:.4f}")

print("\nCoefficients:")
print(f"Linear weight: {lin_model.linear.weight.detach().numpy()}")
print(f"Linear bias: {lin_model.linear.bias.item()}")

# Show fitted model
print(f"Fitted model is approximately: λ(s) = exp({lin_model.linear.bias.item():.4f} + {lin_model.linear.weight.detach().numpy()[0,0]:.4f}*x + {lin_model.linear.weight.detach().numpy()[0,1]:.4f}*y)")



# Plots
import matplotlib.pyplot as plt

# Plot observed point pattern
plt.figure(figsize=(8, 6))

plt.scatter(
    coords_np[:, 0],
    coords_np[:, 1],
    s=10,
    color="darkgreen",
    alpha=0.5,
    label="Longleaf Pines"
)

plt.title("Longleaf Pine Locations in Observation Window")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(alpha=0.3)

plt.legend(
    bbox_to_anchor=(0.0, -0.15),
    loc="lower left"
)

plt.tight_layout()
plt.show()

# Plot fitted intensity surface for Linear IPPP compared to HPPP baseline
def predict_intensity_surface(model, n_per_dim=200):
    xs = np.linspace(x_min, x_max, n_per_dim)
    ys = np.linspace(y_min, y_max, n_per_dim)

    gx, gy = np.meshgrid(xs, ys)

    grid_raw = np.column_stack([
        gx.ravel(),
        gy.ravel()
    ])

    grid_norm = (grid_raw - mean) / std
    grid_tensor = torch.tensor(grid_norm, dtype=torch.float32)

    with torch.no_grad():
        intensity = model(grid_tensor).numpy().reshape(n_per_dim, n_per_dim)

    return gx, gy, intensity


gx, gy, intensity = predict_intensity_surface(lin_model, n_per_dim=200)
hppp_surface = np.full_like(intensity, lambda_hat)

# Shared color scale across both models
vmin = min(hppp_surface.min(), intensity.min())
vmax = max(hppp_surface.max(), intensity.max())
levels = np.linspace(vmin, vmax, 30)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6),
    sharex=True,
    sharey=True,
    constrained_layout=True
)

# HPPP
im0 = axes[0].contourf(
    gx,
    gy,
    hppp_surface,
    levels=levels,
    vmin=vmin,
    vmax=vmax
)

axes[0].scatter(
    coords_np[:, 0],
    coords_np[:, 1],
    s=8,
    color="black",
    alpha=0.35
)

axes[0].set_title("HPPP: Constant Intensity")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].grid(alpha=0.2)

# Linear IPPP
im1 = axes[1].contourf(
    gx,
    gy,
    intensity,
    levels=levels,
    vmin=vmin,
    vmax=vmax
)

axes[1].scatter(
    coords_np[:, 0],
    coords_np[:, 1],
    s=8,
    color="black",
    alpha=0.35
)

axes[1].set_title("Linear IPPP: Fitted Intensity")
axes[1].set_xlabel("x")
axes[1].grid(alpha=0.2)

# Shared colorbar outside both panels
cbar = fig.colorbar(
    im1,
    ax=axes,
    location="right",
    shrink=0.85,
    pad=0.03
)

cbar.set_label("Estimated intensity λ(s)")

plt.show()