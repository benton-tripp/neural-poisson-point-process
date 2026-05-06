# Neural Poisson Point Process

Neural Inhomogeneous Poisson Point Process implementation in PyTorch.

## Background

This code models the spatial distribution of Longleaf pine trees using Poisson point process models.

The dataset contains tree locations and diameter-at-breast-height marks for Longleaf pine trees in a rectangular observation window in southern Georgia. In the current implementation, only the spatial coordinates are modeled. The DBH marks are retained in the dataset but are not yet used.

The goal is to compare a homogeneous spatial point process against a simple inhomogeneous model that allows tree intensity to vary over space.

## Point Process Framework

Let $W$ denote the observation window and let $s_i = (x_i, y_i)$ denote observed tree locations.

An inhomogeneous Poisson point process is defined by an intensity function:

$$
\lambda(s) \geq 0
$$

where $\lambda(s)$ gives the expected density of events near location $s$.

The IPPP log-likelihood is:

$$
\log L(\lambda)=\sum_{i=1}^{n} \log \lambda(s_i)-\int_W \lambda(u)\,du
$$

The first term rewards high intensity at observed tree locations.  
The second term penalizes total expected intensity over the observation window.

## Assumptions

The implementation assumes:

- Nonnegative intensity: $\lambda(s) \geq 0$
- Independent increments across disjoint spatial regions
- A finite expected number of points over the observation window
- No temporal component
- No interaction between points beyond first-order intensity variation

The current model is therefore a first-order intensity model, not a clustering, inhibition, or interaction model.

## Berman-Turner Quadrature Approximation

Rather than directly optimizing the continuous IPPP likelihood, the code uses a Berman-Turner quadrature approximation.

The observation window is divided into grid cells. For each cell:

$$
Y_j \sim \text{Poisson}(w_j \lambda(u_j))
$$

where:

- $Y_j$ is the number of observed points in grid cell $j$
- $u_j$ is the center of grid cell $j$
- $w_j$ is the area of grid cell $j$
- $\lambda(u_j)$ is the fitted intensity at the cell center

The approximate log-likelihood used for fitting is:

$$
\sum_j \left[Y_j \log \lambda(u_j) - w_j \lambda(u_j) \right]
$$

Constants independent of model parameters are omitted.

This converts the IPPP estimation problem into a weighted Poisson-regression-like problem.

## Coordinate Preprocessing

The observed coordinates are standardized before being passed into the PyTorch models:

$$
x^* = \frac{x - \bar{x}}{s_x},
\quad
y^* = \frac{y - \bar{y}}{s_y}
$$

The same transformation is applied to the Berman-Turner grid cell centers and to prediction grids used for plotting.

This improves numerical stability and makes the linear coefficients correspond to standardized coordinate effects.

## Models

### 1. Homogeneous Poisson Point Process

The HPPP assumes constant intensity over the whole observation window:

$$
\lambda(s) = \lambda_0
$$

The maximum likelihood estimate is available in closed form:

$$
\hat{\lambda}_0 = \frac{n}{|W|}
$$

This model serves as the primary baseline.

### 2. Constant Neural IPPP

The constant neural model is a PyTorch version of the HPPP:

$$
\lambda(s) = \exp(\theta)
$$

It has one learnable parameter and is initialized at the HPPP estimate. It is mainly used as a sanity check.

If implemented correctly, its fitted log-likelihood should closely match the closed-form HPPP log-likelihood.

### 3. Linear IPPP

The linear IPPP allows first-order spatial variation in intensity:

$$
\lambda(s) = \exp(\beta_0 + \beta_1 x^* + \beta_2 y^*)
$$

This is a log-linear inhomogeneous Poisson point process.

It is initialized at the HPPP solution by setting:

$$
\beta_1 = 0,
\quad
\beta_2 = 0,
\quad
\beta_0 = \log(\hat{\lambda}_0)
$$

Training then estimates whether a linear spatial trend improves fit over the homogeneous baseline.

## Optimization

The PyTorch models are optimized with Adam using the negative Berman-Turner log-likelihood.

For numerical stability, the linear predictor is clamped inside the model's forward pass before exponentiation. This clamp is active during both training and prediction.

$$
\eta = \beta_0 + \beta_1 x^* + \beta_2 y^*
$$

$$
\lambda(s) = \exp(\text{clamp}(\eta))
$$

This prevents extreme intensity values during optimization.

## Model Comparison

The fitted models are compared using:

1. **Log-likelihood**: Higher log-likelihood indicates better fit.
2. **AIC**: $AIC = 2k - 2\log L$, where $k$ is the number of fitted parameters.
3. **BIC**: $BIC = k \log(n) - 2\log L$, where $n$ is the number of observed points.
4. **Likelihood Ratio Test**: The linear IPPP is compared against the HPPP using: $LR = 2(\log L_{\text{linear}} - \log L_{\text{HPPP}})$. Because the linear model adds two parameters, the test uses $df = 2$. The p-value is computed from a chi-square distribution.

## Current Interpretation

The constant neural model recovers the closed-form HPPP likelihood, validating the Berman-Turner implementation.

The linear IPPP improves the log-likelihood and reduces both AIC and BIC, suggesting evidence of first-order spatial inhomogeneity.

The fitted coefficients indicate that intensity varies primarily along the standardized y-axis, with a much weaker x-axis effect.

## Current Limitations

The current implementation does not yet model:

- DBH marks
- Nonlinear intensity surfaces
- Spatial interaction between trees
- Clustering or inhibition
- Covariates such as soil, elevation, or environmental raster data
- Residual diagnostics such as quadrat residuals or inhomogeneous K-functions

## Results

The homogeneous Poisson point process (HPPP), constant neural IPPP, and linear IPPP were fit and compared using the Berman-Turner approximate log-likelihood.

| Model | k | BT Log-Likelihood | AIC | BIC |
|---|---:|---:|---:|---:|
| HPPP | 1 | -3052.4124 | 6106.8247 | 6111.1946 |
| Constant NN | 1 | -3052.4126 | 6106.8252 | 6111.1951 |
| Linear IPPP | 3 | -3035.2646 | 6076.5293 | 6089.6390 |

The constant neural model closely matches the closed-form HPPP likelihood. This provides a sanity check that the Berman-Turner likelihood and PyTorch optimization are behaving as expected.

The linear IPPP improves the Berman-Turner log-likelihood relative to the HPPP:

$$
\Delta \log L = -3035.2646 - (-3052.4124) = 17.1478
$$

Both AIC and BIC are lower for the linear IPPP, indicating that the improvement in likelihood is large enough to justify the two additional spatial trend parameters.

### Likelihood Ratio Test

The linear IPPP was compared against the HPPP using a likelihood ratio test:

$$
LR = 2(\log L_{\text{linear}} - \log L_{\text{HPPP}})
$$

| Comparison | df | LR Statistic | p-value |
|---|---:|---:|---:|
| Linear IPPP vs HPPP | 2 | 34.2954 | 3.57146e-08 |

The likelihood ratio test strongly rejects the homogeneous intensity model. This suggests evidence of first-order spatial inhomogeneity in the Longleaf pine point pattern.

### Fitted Linear IPPP

The fitted linear IPPP coefficients were:

| Parameter | Estimate |
|---|---:|
| Intercept | -4.1976 |
| x coefficient | -0.0205 |
| y coefficient | 0.2100 |

Using standardized coordinates, the fitted intensity model is approximately:

$$
\lambda(s) = \exp(-4.1976 - 0.0205x^* + 0.2100y^*)
$$

The x-direction effect is small and slightly negative. The y-direction effect is larger and positive, indicating that fitted intensity increases primarily along the positive standardized y-axis.

### Visual Diagnostics

The observed point pattern is shown below.

![A scatterplot of observed tree locations](images/longleaf_point_pattern.png)

The fitted intensity comparison shows the HPPP constant intensity surface against the linear IPPP fitted intensity surface. Both panels use a shared color scale.

![A side-by-side comparison of HPPP constant intensity surface and Linear IPPP fitted intensity surface](images/longleaf_intensity_comparison.png)

## Planned Extensions

Potential next steps include:

- Adding a nonlinear neural network intensity model
- Comparing linear and nonlinear IPPPs with AIC, BIC, and held-out likelihood
- Modeling marks jointly or conditionally on location
- Adding spatial residual diagnostics
- Testing sensitivity to Berman-Turner grid resolution
- Incorporating spatial covariates