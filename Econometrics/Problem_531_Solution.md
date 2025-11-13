# Econometric Problem 531 - Solution

## Problem Statement
Problem 531: Time Series Analysis and Autoregressive Models

Consider an AR(1) process: Y_t = φY_{t-1} + ε_t, where ε_t ~ N(0, σ²)

Tasks:
1. Determine the conditions for stationarity
2. Derive the variance of Y_t under stationarity
3. Calculate the autocorrelation function (ACF)
4. Discuss the estimation of φ using OLS

## Solution

### Part 1: Conditions for Stationarity

For an AR(1) process to be stationary, the following condition must hold:

**Condition: |φ| < 1**

This ensures that:
- The process does not explode over time
- The mean and variance remain constant
- The autocovariance depends only on the lag, not on time

**Proof:**
The AR(1) process can be rewritten as:
```
Y_t = φY_{t-1} + ε_t
    = φ(φY_{t-2} + ε_{t-1}) + ε_t
    = φ²Y_{t-2} + φε_{t-1} + ε_t
    = ... (by backward substitution)
    = Σ(i=0 to ∞) φⁱε_{t-i}
```

This infinite sum converges if and only if |φ| < 1.

### Part 2: Variance under Stationarity

Under stationarity, E[Y_t] = 0 (assuming mean-zero process)

For variance:
```
Var(Y_t) = E[Y_t²]
         = E[(φY_{t-1} + ε_t)²]
         = E[φ²Y_{t-1}² + 2φY_{t-1}ε_t + ε_t²]
         = φ²E[Y_{t-1}²] + E[ε_t²]    (since E[Y_{t-1}ε_t] = 0)
         = φ²Var(Y_{t-1}) + σ²
```

Under stationarity, Var(Y_t) = Var(Y_{t-1}) = γ₀, so:
```
γ₀ = φ²γ₀ + σ²
γ₀(1 - φ²) = σ²
```

**Result: γ₀ = σ²/(1 - φ²)**

This is valid only when |φ| < 1.

### Part 3: Autocorrelation Function (ACF)

The autocovariance at lag k is:
```
γₖ = Cov(Y_t, Y_{t-k})
   = E[Y_tY_{t-k}]    (assuming mean zero)
```

For k = 1:
```
γ₁ = E[Y_tY_{t-1}]
   = E[(φY_{t-1} + ε_t)Y_{t-1}]
   = φE[Y_{t-1}²] + E[ε_tY_{t-1}]
   = φγ₀    (since ε_t is independent of Y_{t-1})
```

By similar reasoning:
```
γₖ = φγ_{k-1} = φᵏγ₀
```

The autocorrelation function is:
```
ρₖ = γₖ/γ₀ = φᵏ
```

**Result: ρₖ = φᵏ**

Properties:
- ρ₀ = 1 (perfect correlation with itself)
- ρₖ decays exponentially as k increases
- The sign of φ determines whether the decay is monotonic (φ > 0) or oscillating (φ < 0)

### Part 4: OLS Estimation

To estimate φ, we use OLS on the regression:
```
Y_t = φY_{t-1} + ε_t
```

**OLS Estimator:**
```
φ̂ = Σ(t=2 to T) Y_tY_{t-1} / Σ(t=2 to T) Y²_{t-1}
```

**Properties of φ̂:**

1. **Consistency:** As T → ∞, φ̂ →ᵖ φ (converges in probability to true value)

2. **Asymptotic Distribution:**
   ```
   √T(φ̂ - φ) →ᵈ N(0, 1 - φ²)
   ```

3. **Bias in Small Samples:** 
   - φ̂ is biased downward in small samples (Hurwicz bias)
   - Bias ≈ -(1 + 3φ)/T for large φ

4. **Standard Error:**
   ```
   SE(φ̂) ≈ √[(1 - φ̂²)/T]
   ```

**Hypothesis Testing:**

To test H₀: φ = 0 (no autocorrelation) vs H₁: φ ≠ 0:
```
t = φ̂/SE(φ̂) ~ N(0,1) approximately for large T
```

To test H₀: φ = 1 (unit root, non-stationarity):
Use Dickey-Fuller test rather than standard t-test.

## Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def simulate_ar1(phi, sigma, n, y0=0):
    """
    Simulate an AR(1) process
    
    Parameters:
    phi: AR(1) coefficient
    sigma: standard deviation of innovations
    n: number of observations
    y0: initial value
    """
    y = np.zeros(n)
    y[0] = y0
    epsilon = np.random.normal(0, sigma, n)
    
    for t in range(1, n):
        y[t] = phi * y[t-1] + epsilon[t]
    
    return y

def estimate_ar1(y):
    """
    Estimate AR(1) coefficient using OLS
    
    Parameters:
    y: time series array
    
    Returns:
    phi_hat: estimated coefficient
    se: standard error
    """
    y_lag = y[:-1]
    y_current = y[1:]
    
    phi_hat = np.sum(y_current * y_lag) / np.sum(y_lag**2)
    
    # Calculate residuals
    residuals = y_current - phi_hat * y_lag
    sigma2_hat = np.sum(residuals**2) / (len(residuals) - 1)
    
    # Standard error
    se = np.sqrt(sigma2_hat / np.sum(y_lag**2))
    
    return phi_hat, se

def calculate_acf(y, max_lag=20):
    """
    Calculate sample autocorrelation function
    
    Parameters:
    y: time series array
    max_lag: maximum lag to compute
    
    Returns:
    acf: array of autocorrelations
    """
    y_centered = y - np.mean(y)
    n = len(y)
    acf = np.zeros(max_lag + 1)
    
    variance = np.sum(y_centered**2) / n
    
    for k in range(max_lag + 1):
        if k == 0:
            acf[k] = 1.0
        else:
            covariance = np.sum(y_centered[k:] * y_centered[:-k]) / n
            acf[k] = covariance / variance
    
    return acf

# Example usage
if __name__ == "__main__":
    # Set parameters
    phi_true = 0.7
    sigma = 1.0
    n = 1000
    
    # Simulate AR(1) process
    np.random.seed(531)  # Problem number as seed
    y = simulate_ar1(phi_true, sigma, n)
    
    # Estimate coefficient
    phi_hat, se = estimate_ar1(y)
    
    # Calculate theoretical values
    theoretical_variance = sigma**2 / (1 - phi_true**2)
    sample_variance = np.var(y)
    
    print(f"Problem 531 Solution Results")
    print(f"=" * 50)
    print(f"True φ: {phi_true}")
    print(f"Estimated φ: {phi_hat:.4f}")
    print(f"Standard Error: {se:.4f}")
    print(f"t-statistic: {phi_hat/se:.4f}")
    print(f"")
    print(f"Theoretical Variance: {theoretical_variance:.4f}")
    print(f"Sample Variance: {sample_variance:.4f}")
    print(f"")
    
    # Test stationarity condition
    if abs(phi_hat) < 1:
        print(f"✓ Stationarity condition satisfied: |φ̂| = {abs(phi_hat):.4f} < 1")
    else:
        print(f"✗ Stationarity condition violated: |φ̂| = {abs(phi_hat):.4f} ≥ 1")
    
    # Calculate and display ACF
    acf = calculate_acf(y, max_lag=20)
    theoretical_acf = phi_true ** np.arange(21)
    
    print(f"\nAutocorrelation Function (first 5 lags):")
    print(f"Lag\tSample ACF\tTheoretical ACF")
    for k in range(6):
        print(f"{k}\t{acf[k]:.4f}\t\t{theoretical_acf[k]:.4f}")
```

## R Implementation

```r
# Problem 531: AR(1) Process Analysis

# Function to simulate AR(1) process
simulate_ar1 <- function(phi, sigma, n, y0 = 0) {
  y <- numeric(n)
  y[1] <- y0
  epsilon <- rnorm(n, mean = 0, sd = sigma)
  
  for (t in 2:n) {
    y[t] <- phi * y[t-1] + epsilon[t]
  }
  
  return(y)
}

# Function to estimate AR(1) coefficient
estimate_ar1 <- function(y) {
  n <- length(y)
  y_lag <- y[1:(n-1)]
  y_current <- y[2:n]
  
  phi_hat <- sum(y_current * y_lag) / sum(y_lag^2)
  
  # Calculate standard error
  residuals <- y_current - phi_hat * y_lag
  sigma2_hat <- sum(residuals^2) / (length(residuals) - 1)
  se <- sqrt(sigma2_hat / sum(y_lag^2))
  
  return(list(phi_hat = phi_hat, se = se))
}

# Set parameters
phi_true <- 0.7
sigma <- 1.0
n <- 1000

# Simulate
set.seed(531)
y <- simulate_ar1(phi_true, sigma, n)

# Estimate
result <- estimate_ar1(y)
phi_hat <- result$phi_hat
se <- result$se

# Calculate theoretical values
theoretical_var <- sigma^2 / (1 - phi_true^2)
sample_var <- var(y)

# Print results
cat("Problem 531 Solution Results\n")
cat(strrep("=", 50), "\n")
cat(sprintf("True φ: %.4f\n", phi_true))
cat(sprintf("Estimated φ: %.4f\n", phi_hat))
cat(sprintf("Standard Error: %.4f\n", se))
cat(sprintf("t-statistic: %.4f\n", phi_hat/se))
cat("\n")
cat(sprintf("Theoretical Variance: %.4f\n", theoretical_var))
cat(sprintf("Sample Variance: %.4f\n", sample_var))
cat("\n")

# Test stationarity
if (abs(phi_hat) < 1) {
  cat(sprintf("✓ Stationarity condition satisfied: |φ̂| = %.4f < 1\n", abs(phi_hat)))
} else {
  cat(sprintf("✗ Stationarity condition violated: |φ̂| = %.4f ≥ 1\n", abs(phi_hat)))
}

# ACF analysis
acf_result <- acf(y, lag.max = 20, plot = FALSE)
cat("\nSample ACF (first 5 lags):\n")
print(acf_result$acf[1:6])
```

## Summary

This solution demonstrates:
1. **Stationarity condition**: |φ| < 1 is necessary and sufficient
2. **Variance formula**: γ₀ = σ²/(1 - φ²) under stationarity
3. **ACF**: ρₖ = φᵏ, showing exponential decay
4. **OLS estimation**: Consistent and asymptotically normal estimator

The AR(1) model is fundamental in time series econometrics and forms the basis for more complex ARMA and ARIMA models. Understanding its properties is crucial for:
- Business cycle analysis
- Financial time series modeling
- Forecasting economic variables
- Testing for unit roots and cointegration

## References

1. Hamilton, J. D. (1994). Time Series Analysis. Princeton University Press.
2. Hayashi, F. (2000). Econometrics. Princeton University Press.
3. Greene, W. H. (2018). Econometric Analysis (8th ed.). Pearson.
4. Wooldridge, J. M. (2015). Introductory Econometrics: A Modern Approach (6th ed.). Cengage Learning.
