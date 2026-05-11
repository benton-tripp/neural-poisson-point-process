# Import packages
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import chi2
import matplotlib.pyplot as plt

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
# Mark preprocessing
# ---------------------------------------------------
# Use log(DBH) as the mark response for numerical stability.
marks_np = df["marks"].values.astype(np.float32)
log_marks_np = np.log(marks_np)

log_marks_mean = log_marks_np.mean()
log_marks_std = log_marks_np.std()

log_marks_norm = (log_marks_np - log_marks_mean) / log_marks_std
log_marks = torch.tensor(log_marks_norm[:, None], dtype=torch.float32)

coords_obs = torch.tensor((coords_np - mean) / std, dtype=torch.float32)

# ---------------------------------------------------
# Models
# ---------------------------------------------------

# Constant IPPP model
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

# Linear IPPP model
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
    
# Conditional mark model
class LinearGaussianMarkModel(nn.Module):
    """
    Linear Gaussian mark model for log(DBH).

    The conditional mean is:

        mu(s) = alpha0 + alpha1 x* + alpha2 y*

    and the conditional variance is estimated through log_sigma.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        mu = self.linear(x)
        sigma = torch.exp(self.log_sigma)
        return mu, sigma

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

# Mark log-likelihood for the conditional Gaussian mark model
def mark_loglik(model, x, y):
    mu, sigma = model(x)

    return (
        -torch.log(sigma + 1e-8)
        -0.5 * ((y - mu) / (sigma + 1e-8)) ** 2
        -0.5 * np.log(2 * np.pi)
    ).sum().item()

# Mark negative log-likelihood for optimization, for the conditional Gaussian mark model
def mark_nll(model, x, y):
    mu, sigma = model(x)

    return -(
        -torch.log(sigma + 1e-8)
        -0.5 * ((y - mu) / (sigma + 1e-8)) ** 2
        -0.5 * np.log(2 * np.pi)
    ).sum()

# General Berman-Turner likelihood for arbitrary grid
def bt_loglik_grid(model, q_coords, q_counts, q_weights):
    lambda_vals = model(q_coords)

    return (
        q_counts * torch.log(lambda_vals + 1e-8)
        - q_weights * lambda_vals
    ).sum().item()


def bt_nll_grid(model, q_coords, q_counts, q_weights):
    lambda_vals = model(q_coords)

    return -(
        q_counts * torch.log(lambda_vals + 1e-8)
        - q_weights * lambda_vals
    ).sum()

# ---------------------------------------------------
# Fit helpers
# ---------------------------------------------------

# Fit the spatial IPPP model using the Berman-Turner negative log-likelihood.
def fit(model, epochs=2000, lr=1e-3):
    opt = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        opt.zero_grad()
        loss = bt_nll(model)
        loss.backward()
        opt.step()

    return model

# Fit the conditional mark model using the mark negative log-likelihood.
def fit_mark_model(model, x, y, epochs=2000, lr=1e-3):
    opt = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        opt.zero_grad()
        loss = mark_nll(model, x, y)
        loss.backward()
        opt.step()

    return model

# For fitting on an arbitrary grid (e.g. for spatial residual diagnostics)
def fit_grid(model, q_coords, q_counts, q_weights, epochs=2000, lr=1e-3):
    opt = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        opt.zero_grad()
        loss = bt_nll_grid(model, q_coords, q_counts, q_weights)
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

# Model 4: Conditional mark model
# This models the DBH mark conditional on observed tree location:
#
#   log(mark_i) | s_i ~ Normal(mu(s_i), sigma^2)
#
# This is not changing the spatial IPPP intensity λ(s).
# It adds a conditional mark distribution p(m | s).
mark_model = fit_mark_model(
    LinearGaussianMarkModel(input_dim=2),
    coords_obs,
    log_marks,
    epochs=3000,
    lr=1e-3
)
mark_ll = mark_loglik(mark_model, coords_obs, log_marks)

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

print("\nConditional Mark Model:")
print(f"Mark log-likelihood: {mark_ll:.4f}")
print(f"Mark coefficients: {mark_model.linear.weight.detach().numpy()}")
print(f"Mark intercept: {mark_model.linear.bias.item():.4f}")
print(f"Mark sigma: {torch.exp(mark_model.log_sigma).item():.4f}")

joint_loglik_linear_marked = loglik_lin + mark_ll
print(f"Joint spatial + mark log-likelihood: {joint_loglik_linear_marked:.4f}")

# ---------------------------------------------------
# Spatial residual diagnostics
# ---------------------------------------------------
def berman_turner_residuals(model):
    with torch.no_grad():
        lambda_vals = model(quad_coords)
        expected_counts = quad_weights * lambda_vals

        raw_resid = quad_counts - expected_counts
        pearson_resid = raw_resid / torch.sqrt(expected_counts + 1e-8)

    return (
        raw_resid.numpy().ravel(),
        pearson_resid.numpy().ravel(),
        expected_counts.numpy().ravel(),
        quad_counts.numpy().ravel()
    )


raw_resid, pearson_resid, expected_counts, observed_counts = berman_turner_residuals(lin_model)

print("\nSpatial Residual Diagnostics:")
print(f"Mean raw residual: {raw_resid.mean():.6f}")
print(f"Mean Pearson residual: {pearson_resid.mean():.6f}")
print(f"Pearson residual SD: {pearson_resid.std():.6f}")
print(f"Observed total count: {observed_counts.sum():.0f}")
print(f"Expected total count: {expected_counts.sum():.4f}")

# ---------------------------------------------------
# Sensitivity to Berman-Turner grid resolution
# ---------------------------------------------------
def run_grid_sensitivity(grid_sizes=(50, 75, 100, 150, 200)):
    rows = []

    for g in grid_sizes:
        q_coords, q_counts, q_weights = make_berman_turner_grid(n_per_dim=g)

        # HPPP is closed-form and does not depend on grid resolution
        ll_hppp = loglik_hppp

        # Constant NN
        const_g = fit_grid(
            ConstantIPPP(lambda_hat),
            q_coords,
            q_counts,
            q_weights,
            epochs=1000,
            lr=1e-3
        )
        ll_const = bt_loglik_grid(const_g, q_coords, q_counts, q_weights)

        # Linear IPPP
        lin_g = fit_grid(
            LinearIPPP(input_dim=2, lambda_init=lambda_hat),
            q_coords,
            q_counts,
            q_weights,
            epochs=3000,
            lr=1e-3
        )
        ll_lin = bt_loglik_grid(lin_g, q_coords, q_counts, q_weights)

        lr = lr_stat(ll_lin, ll_hppp)
        p_value = chi2.sf(lr, df=2)

        rows.append({
            "grid_n_per_dim": g,
            "n_cells": g * g,
            "HPPP_BT_LogLik": ll_hppp,
            "Constant_BT_LogLik": ll_const,
            "Linear_BT_LogLik": ll_lin,
            "Linear_AIC": aic(ll_lin, 3),
            "Linear_BIC": bic(ll_lin, 3, n),
            "LR_Linear_vs_HPPP": lr,
            "LR_p_value": p_value,
            "beta_x": lin_g.linear.weight.detach().numpy()[0, 0],
            "beta_y": lin_g.linear.weight.detach().numpy()[0, 1],
            "beta_0": lin_g.linear.bias.item()
        })

    return pd.DataFrame(rows)


grid_sensitivity_df = run_grid_sensitivity(
    grid_sizes=(50, 75, 100, 150, 200)
)

print("\nBerman-Turner Grid Sensitivity:")
print(grid_sensitivity_df)

# Plots

# Plot observed point pattern
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    coords_np[:, 0],
    coords_np[:, 1],
    s=10,
    color="darkgreen",
    alpha=0.5,
    label="Longleaf Pines"
)

ax.set_title("Longleaf Pine Locations in Observation Window")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.grid(alpha=0.3)

ax.legend(
    bbox_to_anchor=(0.0, -0.15),
    loc="lower left"
)

fig.tight_layout()

# Save before showing
fig.savefig(
    "images/longleaf_point_pattern.png",
    dpi=300,
    bbox_inches="tight"
)

# plt.show()
# Save plot
plt.close(fig)

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

fig.savefig(
    "images/longleaf_intensity_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
# plt.show()
plt.close(fig)

# Plot Pearson residual surface
n_per_dim = 100
resid_grid = pearson_resid.reshape(n_per_dim, n_per_dim)

x_edges = np.linspace(x_min, x_max, n_per_dim + 1)
y_edges = np.linspace(y_min, y_max, n_per_dim + 1)

x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

gx_resid, gy_resid = np.meshgrid(x_centers, y_centers, indexing="ij")

vmax = np.nanmax(np.abs(resid_grid))
vmin = -vmax

fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

im = ax.contourf(
    gx_resid,
    gy_resid,
    resid_grid,
    levels=np.linspace(vmin, vmax, 31),
    vmin=vmin,
    vmax=vmax
)

ax.scatter(
    coords_np[:, 0],
    coords_np[:, 1],
    s=6,
    color="black",
    alpha=0.25
)

ax.set_title("Pearson Residual Surface: Linear IPPP")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.grid(alpha=0.2)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Pearson residual")

fig.savefig(
    "images/linear_ippp_pearson_residuals.png",
    dpi=300,
    bbox_inches="tight"
)

# plt.show()
plt.close(fig)

# Plot grid sensitivity
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(
    grid_sensitivity_df["grid_n_per_dim"],
    grid_sensitivity_df["Linear_BT_LogLik"],
    marker="o",
    label="Linear IPPP"
)

ax.axhline(
    loglik_hppp,
    linestyle="--",
    label="HPPP"
)

ax.set_title("Sensitivity to Berman-Turner Grid Resolution")
ax.set_xlabel("Grid cells per dimension")
ax.set_ylabel("BT log-likelihood")
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(
    "images/grid_sensitivity_loglik.png",
    dpi=300,
    bbox_inches="tight"
)

# plt.show()
plt.close(fig)