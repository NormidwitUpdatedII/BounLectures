"""
Econometric Problem 531 - AR(1) Process Analysis
Author: BounLectures
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def simulate_ar1(phi, sigma, n, y0=0):
    """
    Simulate an AR(1) process: Y_t = φY_{t-1} + ε_t
    
    Parameters:
    -----------
    phi : float
        AR(1) coefficient (must satisfy |phi| < 1 for stationarity)
    sigma : float
        Standard deviation of innovations
    n : int
        Number of observations
    y0 : float
        Initial value (default: 0)
    
    Returns:
    --------
    y : ndarray
        Simulated time series
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
    -----------
    y : ndarray
        Time series array
    
    Returns:
    --------
    phi_hat : float
        Estimated AR(1) coefficient
    se : float
        Standard error of the estimate
    """
    y_lag = y[:-1]
    y_current = y[1:]
    
    # OLS estimate: φ̂ = Σ(Y_t * Y_{t-1}) / Σ(Y²_{t-1})
    phi_hat = np.sum(y_current * y_lag) / np.sum(y_lag**2)
    
    # Calculate residuals
    residuals = y_current - phi_hat * y_lag
    sigma2_hat = np.sum(residuals**2) / (len(residuals) - 1)
    
    # Standard error: SE(φ̂) ≈ √(σ²/Σ(Y²_{t-1}))
    se = np.sqrt(sigma2_hat / np.sum(y_lag**2))
    
    return phi_hat, se


def calculate_acf(y, max_lag=20):
    """
    Calculate sample autocorrelation function
    
    Parameters:
    -----------
    y : ndarray
        Time series array
    max_lag : int
        Maximum lag to compute
    
    Returns:
    --------
    acf : ndarray
        Array of autocorrelations
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


def main():
    """
    Main function to demonstrate Problem 531 solution
    """
    print("="*60)
    print("ECONOMETRIC PROBLEM 531 - AR(1) PROCESS ANALYSIS")
    print("="*60)
    print()
    
    # Set parameters
    phi_true = 0.7
    sigma = 1.0
    n = 1000
    
    print("Problem Setup:")
    print(f"  AR(1) Process: Y_t = φY_(t-1) + ε_t")
    print(f"  True φ = {phi_true}")
    print(f"  σ = {sigma}")
    print(f"  Sample size n = {n}")
    print()
    
    # Check stationarity condition
    print("Part 1: Stationarity Condition")
    print(f"  Condition: |φ| < 1")
    print(f"  Check: |{phi_true}| = {abs(phi_true)} < 1")
    if abs(phi_true) < 1:
        print(f"  ✓ Process is stationary")
    else:
        print(f"  ✗ Process is NOT stationary")
    print()
    
    # Simulate AR(1) process
    np.random.seed(531)  # Problem number as seed
    y = simulate_ar1(phi_true, sigma, n)
    
    # Estimate coefficient
    phi_hat, se = estimate_ar1(y)
    
    print("Part 2: Variance under Stationarity")
    # Calculate theoretical values
    theoretical_variance = sigma**2 / (1 - phi_true**2)
    sample_variance = np.var(y)
    print(f"  Theoretical variance: γ₀ = σ²/(1-φ²) = {theoretical_variance:.4f}")
    print(f"  Sample variance:      {sample_variance:.4f}")
    print(f"  Difference:           {abs(theoretical_variance - sample_variance):.4f}")
    print()
    
    print("Part 3: Autocorrelation Function (ACF)")
    # Calculate and display ACF
    acf = calculate_acf(y, max_lag=20)
    theoretical_acf = phi_true ** np.arange(21)
    
    print(f"  Theoretical ACF: ρₖ = φᵏ")
    print()
    print(f"  {'Lag':<6} {'Sample ACF':<14} {'Theoretical ACF':<16} {'Difference':<12}")
    print(f"  {'-'*6} {'-'*14} {'-'*16} {'-'*12}")
    for k in range(min(11, len(acf))):
        diff = abs(acf[k] - theoretical_acf[k])
        print(f"  {k:<6} {acf[k]:<14.4f} {theoretical_acf[k]:<16.4f} {diff:<12.4f}")
    print()
    
    print("Part 4: OLS Estimation")
    print(f"  Estimated coefficient: φ̂ = {phi_hat:.4f}")
    print(f"  True coefficient:      φ = {phi_true:.4f}")
    print(f"  Estimation error:      {abs(phi_hat - phi_true):.4f}")
    print(f"  Standard error:        SE(φ̂) = {se:.4f}")
    print(f"  t-statistic:           t = φ̂/SE(φ̂) = {phi_hat/se:.4f}")
    print()
    
    # Test stationarity of estimated process
    if abs(phi_hat) < 1:
        print(f"  ✓ Estimated process is stationary: |φ̂| = {abs(phi_hat):.4f} < 1")
    else:
        print(f"  ✗ Estimated process is NOT stationary: |φ̂| = {abs(phi_hat):.4f} ≥ 1")
    print()
    
    # Hypothesis test: H₀: φ = 0 vs H₁: φ ≠ 0
    t_stat = phi_hat / se
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    
    print("Hypothesis Test: H₀: φ = 0 vs H₁: φ ≠ 0")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value:     {p_value:.4e}")
    
    if p_value < 0.05:
        print(f"  ✓ Reject H₀ at 5% level: significant autocorrelation detected")
    else:
        print(f"  ✗ Fail to reject H₀ at 5% level")
    print()
    
    # Calculate confidence interval
    ci_lower = phi_hat - 1.96 * se
    ci_upper = phi_hat + 1.96 * se
    print(f"95% Confidence Interval:")
    print(f"  [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    if ci_lower <= phi_true <= ci_upper:
        print(f"  ✓ True value φ = {phi_true} is within the confidence interval")
    else:
        print(f"  ✗ True value φ = {phi_true} is outside the confidence interval")
    print()
    
    print("="*60)
    print("SOLUTION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
