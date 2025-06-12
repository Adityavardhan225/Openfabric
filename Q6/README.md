# Granger Causality Test: Derivation and Adaptation

> **Question:** Given a multivariate time series $\mathbf{X}_t \in \mathbb{R}^n$, derive the Granger causality test in matrix form and describe how you would adapt it for non-stationary financial time series.

---

This repository contains a formal mathematical derivation of the Granger causality test for multivariate time series. It covers the derivation in full matrix form for stationary processes and details the necessary adaptations for non-stationary financial data.

## 1. Derivation in Matrix Form for a Stationary VAR($p$) Process

The derivation proceeds in three steps: (1) defining the VAR($p$) process, (2) stacking the system into a single multivariate linear model, and (3) formulating the test statistic based on this model.

### 1.1 The Vector Autoregressive (VAR) Model
We assume the stationary time series $\mathbf{X}_t \in \mathbb{R}^n$ follows a Vector Autoregressive model of order $p$, denoted VAR($p$):
$$
\mathbf{X}_t = \mathbf{c} + A_1 \mathbf{X}_{t-1} + A_2 \mathbf{X}_{t-2} + \dots + A_p \mathbf{X}_{t-p} + \boldsymbol{\varepsilon}_t
$$
where $\mathbf{c}$ is an $n \times 1$ vector of intercepts, each $A_k$ is an $n \times n$ coefficient matrix, and $\boldsymbol{\varepsilon}_t$ is an $n \times 1$ white noise vector with covariance matrix $\Sigma$.

### 1.2 Stacking the System into Matrix Form
To estimate the system, we stack the $T-p$ observations (from $t=p+1$ to $T$) into a single multivariate linear model of the form $Y = ZB + U$.

-   **Dependent Variables ($Y$):** The observations of $\mathbf{X}_t$ are stacked row-wise.
    $$
    Y = \begin{pmatrix} \mathbf{X}_{p+1}^\top \\ \mathbf{X}_{p+2}^\top \\ \vdots \\ \mathbf{X}_T^\top \end{pmatrix}, \quad \text{a } (T-p) \times n \text{ matrix.}
    $$

-   **Regressors ($Z$):** For each time $t$, we create a row vector containing a constant and all $p$ lagged vectors.
    $$
    Z = \begin{pmatrix}
    1 & \mathbf{X}_{p}^\top & \mathbf{X}_{p-1}^\top & \dots & \mathbf{X}_{1}^\top \\
    1 & \mathbf{X}_{p+1}^\top & \mathbf{X}_{p}^\top & \dots & \mathbf{X}_{2}^\top \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    1 & \mathbf{X}_{T-1}^\top & \mathbf{X}_{T-2}^\top & \dots & \mathbf{X}_{T-p}^\top
    \end{pmatrix}, \quad \text{a } (T-p) \times (1+np) \text{ matrix.}
    $$

-   **Coefficients ($B$):** The coefficient matrices are stacked vertically.
    $$
    B = \begin{pmatrix} \mathbf{c}^\top \\ A_1^\top \\ A_2^\top \\ \vdots \\ A_p^\top \end{pmatrix}, \quad \text{an } (1+np) \times n \text{ matrix.}
    $$

-   **Errors ($U$):** The error vectors are stacked row-wise.
    $$
    U = \begin{pmatrix} \boldsymbol{\varepsilon}_{p+1}^\top \\ \boldsymbol{\varepsilon}_{p+2}^\top \\ \vdots \\ \boldsymbol{\varepsilon}_T^\top \end{pmatrix}, \quad \text{a } (T-p) \times n \text{ matrix.}
    $$

This yields the complete system in matrix form, which can be estimated via Ordinary Least Squares (OLS):
$$
\underbrace{Y}_{(T-p) \times n} = \underbrace{Z}_{(T-p) \times (1+np)} \underbrace{B}_{(1+np) \times n} + \underbrace{U}_{(T-p) \times n}
$$

### 1.3 The Granger Causality Hypothesis and Test Statistic

#### Formulating the Hypothesis
To test if a subset of variables $\mathbf{X}^{(2)}_t \in \mathbb{R}^{n_2}$ Granger-causes another subset $\mathbf{X}^{(1)}_t \in \mathbb{R}^{n_1}$, we examine the partitioned coefficient matrices $A_k$. The null hypothesis states that lagged values of $\mathbf{X}^{(2)}$ do not help predict $\mathbf{X}^{(1)}$.

> **Null Hypothesis ($H_0$):** The coefficients linking past $\mathbf{X}^{(2)}$ to current $\mathbf{X}^{(1)}$ are all zero.
> $$
> H_0: \quad A_{k,12} = \mathbf{0} \quad \text{for all lags } k = 1, \dots, p.
> $$

#### The Likelihood Ratio (LR) Test
We test these $d = n_1 \cdot n_2 \cdot p$ joint linear restrictions by comparing the fit of the unrestricted model with a restricted model where $H_0$ is imposed.

1.  **Unrestricted Model:** Estimate the full system $Y=ZB+U$ via OLS to get $\hat{B}_U$. The residuals are $\hat{U}_U = Y - Z\hat{B}_U$. The residual covariance matrix is $\hat{\Sigma}_U = \frac{1}{T-p} \hat{U}_U^\top \hat{U}_U$.
2.  **Restricted Model:** Estimate the system again, but with the constraints from $H_0$ imposed. This yields the restricted residual covariance matrix $\hat{\Sigma}_R$.

The LR test statistic compares the determinants (generalized variances) of the two residual covariance matrices.

> **Likelihood Ratio (LR) Statistic:**
> $$
>     LR = (T-p) \left( \log|\det(\hat{\Sigma}_R)| - \log|\det(\hat{\Sigma}_U)| \right)
> $$

Under the null hypothesis, this statistic follows a chi-squared distribution asymptotically:
$$
LR \;\xrightarrow{d}\; \chi^2(d), \quad \text{where the degrees of freedom } d = n_1 n_2 p.
$$
A large $LR$ value leads to the rejection of $H_0$, implying Granger causality.

## 2. Adaptation for Non-Stationary Financial Time Series

Financial data (e.g., asset prices) are typically non-stationary, often integrated of order one, I(1). Applying the standard VAR test to I(1) data is invalid. The correct procedure depends on whether the variables are cointegrated.

### Case 1: I(1) Series without Cointegration
If the series are non-stationary but do not share a long-run equilibrium, we must first induce stationarity.
-   **Method:** Transform the data by taking the **first-difference**. For log-prices $\mathbf{p}_t$, this means working with log-returns, $\Delta\mathbf{p}_t = \mathbf{p}_t - \mathbf{p}_{t-1}$.
-   **Procedure:** A VAR model is estimated on the stationary, differenced series, and the Granger causality test is performed as derived above. The interpretation becomes about causality in returns.

### Case 2: I(1) Series with Cointegration
If the series are cointegrated, they share a stable long-run equilibrium. Differencing the data would discard this crucial information.
-   **Method:** Use a **Vector Error Correction Model (VECM)**. A VAR($p$) can be re-written as a VECM:
    $$
        \Delta \mathbf{X}_t = \Pi \mathbf{X}_{t-1} + \sum_{k=1}^{p-1} \Gamma_k \Delta \mathbf{X}_{t-k} + \mathbf{c} + \boldsymbol{\varepsilon}_t
    $$
-   **Causality in a VECM:** Causality now flows from two distinct sources:
    1.  **Short-Run Causality:** Tested via joint zero-restrictions on the coefficients of the $\Gamma_k$ matrices. This is a standard test on the lagged differenced terms.
    2.  **Long-Run Causality:** Tested via the **error correction term** $\Pi = \alpha \beta^\top$. Causality exists if a variable adjusts to past disequilibrium. This is tested by checking for zero-restrictions on the adjustment coefficients in the $\alpha$ matrix. For $\mathbf{X}^{(2)}$ not to cause $\mathbf{X}^{(1)}$, the rows of $\alpha$ corresponding to $\mathbf{X}^{(1)}$ must be zero.

To conclude an absence of Granger causality in a cointegrated system, one must test and fail to reject **both** the absence of short-run and long-run causality.

## 3. Summary Workflow
A robust analysis requires the following steps:
1.  **Test for Unit Roots:** Check all series for non-stationarity (e.g., using an ADF test).
2.  **Test for Cointegration:** If series are I(1), use a Johansen test to find the number of cointegrating relationships.
3.  **Select the Correct Model:**
    -   If I(0) $\implies$ VAR in levels.
    -   If I(1) and not cointegrated $\implies$ VAR in first-differences.
    -   If I(1) and cointegrated $\implies$ VECM.
4.  **Perform Granger Causality Test(s):** Apply the appropriate zero-coefficient hypothesis tests within the chosen model.