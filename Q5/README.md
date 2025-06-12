# Derivation of the Optimal Kelly Fraction with Autocorrelated Gaussian Returns


## Overview

This derivation proceeds in four parts:
1. **The Kelly Criterion Foundation:** Establishing the universal objective function.
2. **The Baseline Case:** Solving for i.i.d. Gaussian returns using a standard approximation.
3. **The Autocorrelated Case:** Introducing an AR(1) model and deriving the dynamic, conditional Kelly fraction.
4. **Analysis and Impact:** Interpreting the result to explain how positive autocorrelation affects optimal leverage.

---

## 1. The Foundation: The Kelly Criterion

The Kelly criterion seeks to maximize the long-run compound growth rate of wealth.

### 1. Wealth Dynamics
Let Wₜ be wealth at time t. If we invest a fraction f of our wealth into an asset with stochastic return rₜ, our wealth evolves as:

Wₜ₊₁ = Wₜ · (1 + f rₜ)

### 2. Objective Function
To maximize the long-run compound growth rate, we maximize the expected logarithm of the wealth growth factor. This is because the total log-wealth after T periods is log Wₜ = log W₀ + ∑ₜ₌₀ᵀ⁻¹ log(1+frₜ). Maximizing the long-run average is equivalent to maximizing the single-period expectation:

G(f) = 𝔼[log(1 + f r)]

### 3. First-Order Condition
To find the optimal fraction f*, we differentiate G(f) with respect to f and set the result to zero. Using Leibniz's rule to differentiate under the expectation:

dG/df = 𝔼[d/df log(1 + f r)] = 𝔼[r/(1 + f r)] = 0

This equation, 𝔼[r/(1 + f r)] = 0, is the universal master equation for the Kelly fraction. For most return distributions, including Gaussian, it has no closed-form solution and requires approximation.

## 2. The Baseline Case: i.i.d. Gaussian Returns

Let's first solve for the simple case where returns are independent and identically distributed (i.i.d.) following a normal distribution, r ~ 𝒩(μ, σ²).

### 1. Taylor Approximation
We approximate the objective function, log(1+x), around x=0 using a second-order Taylor expansion: log(1+x) ≈ x - ½x². Substituting x=fr:

G(f) = 𝔼[log(1+fr)] ≈ 𝔼[fr - ½f²r²]

### 2. Evaluating the Expectation
Using the linearity of expectation:

G(f) ≈ f·𝔼[r] - ½f²·𝔼[r²]

For a random variable X, we know 𝔼[X²] = Var(X) + (𝔼[X])². Therefore, 𝔼[r] = μ and 𝔼[r²] = σ² + μ². Plugging these in, our approximate objective function becomes:

G(f) ≈ fμ - ½f²(σ² + μ²)

### 3. Optimization
We now maximize this quadratic function of f:

dG/df = μ - f(σ² + μ²) = 0

Solving for f gives the optimal fraction:

f* = μ/(σ² + μ²)

**Practical Note:** In most financial applications, the mean return μ is much smaller than the standard deviation σ. This means μ² ≪ σ², leading to the widely used approximation: f* ≈ μ/σ².

## 3. The Autocorrelated Case: The AR(1) Model

We now model the return stream with first-order autocorrelation using an AR(1) process.

### 1. Model Definition
The return at time t is defined as:

rₜ = μ + ρ(rₜ₋₁ - μ) + ηₜ

Where:
- μ is the unconditional long-term mean of the returns
- ρ ∈ (-1, 1) is the lag-1 autocorrelation coefficient
- ηₜ is a Gaussian white noise process (innovation), with ηₜ ~ 𝒩(0, σ²η) and i.i.d. over time

### 2. Conditional Distribution
At time t, when we make our leverage decision, the previous return rₜ₋₁ is known. The optimal decision must be conditioned on this information.

**Conditional Mean:**
μₜ ≡ 𝔼[rₜ | rₜ₋₁] = μ + ρ(rₜ₋₁ - μ)

**Conditional Variance:**
σ²ₜ ≡ Var(rₜ | rₜ₋₁) = σ²η

Therefore, the conditional distribution of the next return is:
rₜ | rₜ₋₁ ~ 𝒩(μ + ρ(rₜ₋₁ - μ), σ²η)

### 3. Deriving the Optimal Dynamic Fraction
We re-apply the result from Part 2 using the conditional parameters. This yields the rigorous, time-varying optimal Kelly fraction:

> fₜ* = (μ + ρ(rₜ₋₁ - μ))/(σ²η + (μ + ρ(rₜ₋₁ - μ))²)

As before, assuming the squared conditional mean is small relative to the conditional variance, this simplifies to the more common, practical formula:

fₜ* ≈ (μ + ρ(rₜ₋₁ - μ))/σ²η

## 4. How Positive Autocorrelation Impacts Optimal Leverage

Positive autocorrelation (ρ > 0) fundamentally changes the optimal strategy and increases the justifiable amount of leverage.

### 1. Leverage Becomes Dynamic and Pro-cyclical
The optimal fraction fₜ* is no longer constant. It adapts based on past performance. If ρ > 0, a positive shock (rₜ₋₁ > μ) increases the conditional mean for the next period, prompting a **higher** allocation.

### 2. Predictability Reduces Effective Risk
The total variance of the process is Var(rₜ) = σ² = σ²η/(1-ρ²). Autocorrelation means a portion of this variance is *predictable*. The relevant risk for a decision is the variance of the unpredictable part—the innovation variance, σ²η. Since σ²η = σ²(1-ρ²), for any ρ ≠ 0, the effective risk is **strictly lower** than the naive, unconditional risk.

### 3. Average Leverage is Higher
We compare the average leverage of the informed strategy to a naive strategy that ignores autocorrelation.

**Naive Strategy (ρ=0):**
f_naive ≈ μ/σ² = μ(1-ρ²)/σ²η

**Informed Strategy:**
The average leverage is 𝔼[fₜ*]:
𝔼[fₜ*] ≈ μ/σ²η

Comparing the two, for any positive autocorrelation (ρ>0):
𝔼[fₜ*] = μ/σ²η > μ(1-ρ²)/σ²η = f_naive

The ability to exploit predictability allows for a higher average leverage.

