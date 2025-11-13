# Econometrics Course Materials

This directory contains solutions and implementations for econometric problems from Bogazici University lectures.

## Contents

### Problem 531: AR(1) Process Analysis

**Topic:** Time Series Analysis and Autoregressive Models

**Files:**
- `Problem_531_Solution.md` - Complete theoretical solution with derivations
- `problem_531.py` - Python implementation
- `problem_531.R` - R implementation

**Problem Overview:**
Analysis of an AR(1) process: Y_t = φY_{t-1} + ε_t, covering:
1. Stationarity conditions
2. Variance derivation
3. Autocorrelation function (ACF)
4. OLS estimation

## Running the Code

### Python Implementation

Requirements:
- Python 3.x
- NumPy
- SciPy
- Matplotlib

```bash
cd Econometrics
pip install numpy scipy matplotlib
python problem_531.py
```

### R Implementation

Requirements:
- R (version 3.x or higher)

```bash
cd Econometrics
Rscript problem_531.R
```

## Expected Output

Both implementations will:
1. Verify stationarity conditions
2. Compare theoretical vs. sample variance
3. Display autocorrelation function
4. Perform OLS estimation with hypothesis tests
5. Show 95% confidence intervals

## Theory Summary

**Stationarity:** An AR(1) process is stationary if |φ| < 1

**Variance:** Under stationarity, γ₀ = σ²/(1 - φ²)

**ACF:** The autocorrelation at lag k is ρₖ = φᵏ

**OLS Estimation:** The OLS estimator is consistent and asymptotically normal:
- φ̂ = Σ(Y_t·Y_{t-1}) / Σ(Y²_{t-1})
- √T(φ̂ - φ) →ᵈ N(0, 1 - φ²)

## References

1. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
2. Hayashi, F. (2000). *Econometrics*. Princeton University Press.
3. Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson.
4. Wooldridge, J. M. (2015). *Introductory Econometrics: A Modern Approach* (6th ed.). Cengage Learning.

## License

These materials are for educational purposes as part of Bogazici University course work.
