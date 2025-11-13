# Econometric Problem 531 - AR(1) Process Analysis
# Author: BounLectures
# Date: November 2025

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

# Function to estimate AR(1) coefficient using OLS
estimate_ar1 <- function(y) {
  n <- length(y)
  y_lag <- y[1:(n-1)]
  y_current <- y[2:n]
  
  # OLS estimate: φ̂ = Σ(Y_t * Y_{t-1}) / Σ(Y²_{t-1})
  phi_hat <- sum(y_current * y_lag) / sum(y_lag^2)
  
  # Calculate residuals and standard error
  residuals <- y_current - phi_hat * y_lag
  sigma2_hat <- sum(residuals^2) / (length(residuals) - 1)
  se <- sqrt(sigma2_hat / sum(y_lag^2))
  
  return(list(phi_hat = phi_hat, se = se))
}

# Main analysis
cat(strrep("=", 60), "\n")
cat("ECONOMETRIC PROBLEM 531 - AR(1) PROCESS ANALYSIS\n")
cat(strrep("=", 60), "\n\n")

# Set parameters
phi_true <- 0.7
sigma <- 1.0
n <- 1000

cat("Problem Setup:\n")
cat(sprintf("  AR(1) Process: Y_t = φY_(t-1) + ε_t\n"))
cat(sprintf("  True φ = %.1f\n", phi_true))
cat(sprintf("  σ = %.1f\n", sigma))
cat(sprintf("  Sample size n = %d\n\n", n))

# Part 1: Stationarity Condition
cat("Part 1: Stationarity Condition\n")
cat(sprintf("  Condition: |φ| < 1\n"))
cat(sprintf("  Check: |%.1f| = %.1f < 1\n", phi_true, abs(phi_true)))
if (abs(phi_true) < 1) {
  cat("  ✓ Process is stationary\n\n")
} else {
  cat("  ✗ Process is NOT stationary\n\n")
}

# Simulate AR(1) process
set.seed(531)  # Problem number as seed
y <- simulate_ar1(phi_true, sigma, n)

# Part 2: Variance under Stationarity
cat("Part 2: Variance under Stationarity\n")
theoretical_var <- sigma^2 / (1 - phi_true^2)
sample_var <- var(y)
cat(sprintf("  Theoretical variance: γ₀ = σ²/(1-φ²) = %.4f\n", theoretical_var))
cat(sprintf("  Sample variance:      %.4f\n", sample_var))
cat(sprintf("  Difference:           %.4f\n\n", abs(theoretical_var - sample_var)))

# Part 3: Autocorrelation Function (ACF)
cat("Part 3: Autocorrelation Function (ACF)\n")
cat("  Theoretical ACF: ρₖ = φᵏ\n\n")

# Calculate ACF
acf_result <- acf(y, lag.max = 20, plot = FALSE)
theoretical_acf <- phi_true ^ (0:20)

cat(sprintf("  %-6s %-14s %-16s %-12s\n", "Lag", "Sample ACF", "Theoretical ACF", "Difference"))
cat(sprintf("  %s %s %s %s\n", strrep("-", 6), strrep("-", 14), strrep("-", 16), strrep("-", 12)))
for (k in 1:11) {
  sample_acf <- acf_result$acf[k]
  theor_acf <- theoretical_acf[k]
  diff <- abs(sample_acf - theor_acf)
  cat(sprintf("  %-6d %-14.4f %-16.4f %-12.4f\n", k-1, sample_acf, theor_acf, diff))
}
cat("\n")

# Part 4: OLS Estimation
result <- estimate_ar1(y)
phi_hat <- result$phi_hat
se <- result$se

cat("Part 4: OLS Estimation\n")
cat(sprintf("  Estimated coefficient: φ̂ = %.4f\n", phi_hat))
cat(sprintf("  True coefficient:      φ = %.4f\n", phi_true))
cat(sprintf("  Estimation error:      %.4f\n", abs(phi_hat - phi_true)))
cat(sprintf("  Standard error:        SE(φ̂) = %.4f\n", se))
cat(sprintf("  t-statistic:           t = φ̂/SE(φ̂) = %.4f\n\n", phi_hat/se))

# Test stationarity of estimated process
if (abs(phi_hat) < 1) {
  cat(sprintf("  ✓ Estimated process is stationary: |φ̂| = %.4f < 1\n\n", abs(phi_hat)))
} else {
  cat(sprintf("  ✗ Estimated process is NOT stationary: |φ̂| = %.4f ≥ 1\n\n", abs(phi_hat)))
}

# Hypothesis test: H₀: φ = 0 vs H₁: φ ≠ 0
t_stat <- phi_hat / se
p_value <- 2 * (1 - pnorm(abs(t_stat)))

cat("Hypothesis Test: H₀: φ = 0 vs H₁: φ ≠ 0\n")
cat(sprintf("  t-statistic: %.4f\n", t_stat))
cat(sprintf("  p-value:     %.4e\n", p_value))

if (p_value < 0.05) {
  cat("  ✓ Reject H₀ at 5% level: significant autocorrelation detected\n\n")
} else {
  cat("  ✗ Fail to reject H₀ at 5% level\n\n")
}

# Calculate confidence interval
ci_lower <- phi_hat - 1.96 * se
ci_upper <- phi_hat + 1.96 * se
cat("95% Confidence Interval:\n")
cat(sprintf("  [%.4f, %.4f]\n", ci_lower, ci_upper))

if (ci_lower <= phi_true && phi_true <= ci_upper) {
  cat(sprintf("  ✓ True value φ = %.1f is within the confidence interval\n\n", phi_true))
} else {
  cat(sprintf("  ✗ True value φ = %.1f is outside the confidence interval\n\n", phi_true))
}

cat(strrep("=", 60), "\n")
cat("SOLUTION COMPLETE\n")
cat(strrep("=", 60), "\n")
