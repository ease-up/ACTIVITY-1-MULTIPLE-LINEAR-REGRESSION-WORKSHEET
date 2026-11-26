# Multiple Linear Regression for Concrete Compressive Strength Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: UCI ML](https://img.shields.io/badge/Dataset-UCI%20Machine%20Learning-orange.svg)](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength)
[![Status: CIE 2 Activity 1 Completed](https://img.shields.io/badge/Status-CIE%202%20Activity%201%20Completed-brightgreen.svg)]()

> **Course:** Regression and Classification Algorithms | Unit I  
> **Activity:** CIE 2 – Activity 1: Multiple Linear Regression  
> **Target:** Concrete Compressive Strength (MPa)  
> **Primary Submission Notebook:** [`activity1_multiple_linear_regression.ipynb`](activity1_multiple_linear_regression.ipynb)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Objective](#objective)
- [Dataset Description and Source](#dataset-description-and-source)
- [Input Features and Target Variable](#input-features-and-target-variable)
- [Libraries and Dependencies](#libraries-and-dependencies)
- [Methodology and Workflow](#methodology-and-workflow)
- [Mathematical Formulation and Manual Calculation](#mathematical-formulation-and-manual-calculation)
- [Model Evaluation and Performance](#model-evaluation-and-performance)
- [Regression Diagnostics and Assumption Assessment](#regression-diagnostics-and-assumption-assessment)
- [Multicollinearity Analysis (VIF)](#multicollinearity-analysis-vif)
- [New Mix Design Prediction](#new-mix-design-prediction)
- [How to Run the Project](#how-to-run-the-project)
  - [Running in Google Colab](#running-in-google-colab)
  - [Local Installation with requirements.txt](#local-installation-with-requirementstxt)
  - [Dataset Download Mechanism](#dataset-download-mechanism)
- [Repository File Structure](#repository-file-structure)
- [Engineering Decision Log and Limitations](#engineering-decision-log-and-limitations)

---

## Project Overview

In civil and structural engineering, **concrete compressive strength** governs structural load capacity, safety, and operational longevity. Conventional laboratory verification requires casting standard test cylinders or cubes and subjecting them to hydraulic compressive crush testing after a standard moist-curing duration (typically 7, 14, 28, or 90 days). 

This empirical curing protocol introduces significant engineering bottlenecks:
1. **Time Delay:** Waiting 28 days for strength confirmation delays construction scheduling and precast manufacturing.
2. **Quality Risks:** Sub-optimal mixture designs detected post-curing result in structural rejection, material wastage, and expensive remediation.
3. **Complex Chemistry:** Modern high-performance concrete integrates multi-component supplementary cementitious materials (blast furnace slag, fly ash) and chemical plasticizers whose non-linear interactions are difficult to model using empirical rule-of-thumb ratios.

This project implements an end-to-end **Multiple Linear Regression (MLR)** framework to predict concrete compressive strength from mix components and curing age, validated through first-principles matrix algebra, diagnostic evaluation, and benchmark comparisons.

---

## Objective

1. Develop a multivariate linear regression model to predict concrete compressive strength (in MPa) based on mixture proportions and curing age.
2. Mathematically derive the Ordinary Least Squares (OLS) closed-form solution via matrix algebra ($\beta = (X^T X)^{-1} X^T y$) on actual dataset observations and verify exact agreement with Scikit-Learn.
3. Compare the full 8-predictor model against an empirical single-feature baseline (`Cement`) and naive mean baseline.
4. Interpret regression coefficients *ceteris paribus* (holding all other variables constant) to verify compliance with physical laws (e.g., Abrams' water-cement ratio law).
5. Evaluate Classical Linear Regression Model (CLRM) assumptions through residual analysis and Variance Inflation Factor (VIF) multicollinearity diagnostics.

---

## Dataset Description and Source

The dataset used in this activity is the benchmark **Concrete Compressive Strength Dataset** contributed by Prof. I-Cheng Yeh (Chung-Hua University) to the **UCI Machine Learning Repository**.

* **Data Source:** [UCI Machine Learning Repository - Concrete Compressive Strength](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength)
* **Direct File URL:** `https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls`
* **Raw Records:** 1,030 experimental formulations
* **Cleaned Records:** 1,005 unique formulations (25 duplicate records removed during data cleaning)
* **Missing Values:** Zero missing or null values across all features

---

## Input Features and Target Variable

The dataset comprises **8 continuous quantitative input features** representing mixture constituent masses per unit volume ($kg/m^3$) and curing duration, and **1 continuous target variable** ($y$):

| Variable Name | Role | Data Type | Units | Physical & Engineering Role |
| :--- | :---: | :---: | :---: | :--- |
| **`Cement`** | Predictor ($X_1$) | Continuous | $kg/m^3$ | Primary hydraulic binder. Hydrates with water to form calcium silicate hydrate (C-S-H) gel, providing primary structural bonding. |
| **`Slag`** | Predictor ($X_2$) | Continuous | $kg/m^3$ | Blast furnace slag; a supplementary cementitious material (SCM) that enhances late-age strength and durability. |
| **`FlyAsh`** | Predictor ($X_3$) | Continuous | $kg/m^3$ | Pozzolanic industrial byproduct. Improves fresh concrete workability and microstructural density while reducing hydration heat. |
| **`Water`** | Predictor ($X_4$) | Continuous | $kg/m^3$ | Chemical hydration agent. Excess water increases mix porosity and reduces hardened compressive strength (Abrams' Law). |
| **`Superplasticizer`** | Predictor ($X_5$) | Continuous | $kg/m^3$ | High-range water-reducing admixture. Disperses cement particles, enabling lower water-binder ratios at high workability. |
| **`CoarseAggregate`** | Predictor ($X_6$) | Continuous | $kg/m^3$ | Crushed stone or gravel ($>4.75 \text{ mm}$). Forms the primary compressive skeletal framework of hardened concrete. |
| **`FineAggregate`** | Predictor ($X_7$) | Continuous | $kg/m^3$ | Natural sand ($<4.75 \text{ mm}$). Fills void spaces between coarse aggregate particles to create a dense matrix. |
| **`Age`** | Predictor ($X_8$) | Continuous | Days | Total moist-curing duration prior to compressive testing (ranging from 1 to 365 days). |
| **`Strength`** | Target ($y$) | Continuous | MPa | Ultimate uniaxial compressive crush resistance tested on standard cylindrical specimens in accordance with ASTM/ISO standards. |

---

## Libraries and Dependencies

The project relies on standard Python data science and machine learning libraries:

| Library | Version Requirement | Purpose in Project |
| :--- | :---: | :--- |
| `numpy` | `*` | Matrix algebra, vector operations, array manipulation |
| `pandas` | `*` | Dataframe ingestion, data cleaning, tabular transformations |
| `matplotlib` | `*` | Subplots, scatter plots, residual diagnostics visualization |
| `seaborn` | `*` | Statistical distributions (KDE) and annotated correlation heatmaps |
| `scikit-learn` | `*` | Train-test splitting, Ordinary Least Squares (`LinearRegression`), evaluation metrics |
| `statsmodels` | `*` | Multicollinearity diagnostics (`variance_inflation_factor`) |
| `xlrd` | `>=2.0.1` | Reading legacy binary Excel (`.xls`, BIFF8) files |
| `urllib` | Standard Library | Automated dataset retrieval from the UCI repository |

---

## Methodology and Workflow

The project is structured into 18 sequential sections implemented in [`activity1_multiple_linear_regression.ipynb`](activity1_multiple_linear_regression.ipynb):

```text
[1. Title & Introduction] 
       │
[2. Problem Understanding]
       │
[3. Dataset Description & Features]
       │
[4. Dataset Loading] ────────► Automated URL retrieval & column standardization
       │
[5. Data Cleaning]   ────────► Deduplication (1,030 -> 1,005 unique samples)
       │
[6. Exploratory Data Analysis] ─► 3x3 distribution grid & correlation heatmap
       │
[7. Manual MLR Calculation] ───► First-principles matrix derivation: β = (X^T X)^-1 X^T y
       │
[8. Train/Test Split] ─────────► 80/20 train-test partition (random_state=42)
       │
[9. Single-Feature Baseline] ──► Univariate OLS model using Cement (R² = 0.2073)
       │
[10. Multiple Linear Regression] ─► Full 8-predictor model parameter estimation
       │
[11. Model Predictions] ──────► Inference on test set (N=201) & Actual vs. Predicted plot
       │
[12. Model Evaluation] ───────► MAE, MSE, RMSE, R², Adjusted R² benchmark comparison
       │
[13. Residual Analysis] ──────► Residuals vs. Fitted scatter & Residual distribution KDE
       │
[14. Multicollinearity (VIF)] ─► Variance Inflation Factor diagnostics (< 10 threshold)
       │
[15. CLRM Assumptions] ───────► Evaluation of all 5 Classical Linear Regression assumptions
       │
[16. New Sample Prediction] ──► Practical inference on standard 28-day concrete mix
       │
[17. Professional Decision Log] ► Documented analytical choices & engineering rationale
       │
[18. Conclusion] ─────────────► Performance summary, limitations, and future directions
```

---

## Mathematical Formulation and Manual Calculation

### 1. Matrix Formulation
Given design matrix $X \in \mathbb{R}^{N \times (p+1)}$ (with leading ones for the intercept) and target vector $y \in \mathbb{R}^{N \times 1}$:

$$X = \begin{bmatrix} 1 & X_{11} & X_{12} \\ 1 & X_{21} & X_{22} \\ \vdots & \vdots & \vdots \\ 1 & X_{N1} & X_{N2} \end{bmatrix}, \quad y = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_N \end{bmatrix}, \quad \beta = \begin{bmatrix} \beta_0 \\ \beta_1 \\ \beta_2 \end{bmatrix}$$

The Ordinary Least Squares objective minimizes the sum of squared residuals $S(\beta) = (y - X\beta)^T(y - X\beta)$. Setting the gradient to zero yields the **Normal Equations**:

$$X^T X \beta = X^T y \implies \beta = (X^T X)^{-1} X^T y$$

### 2. Manual Calculation on 6 Sample Observations (Section 7)
Using rows 5 to 10 with two independent predictors (`Cement` $X_1$, `Water` $X_2$):

* **Design Matrix $X$ ($6 \times 3$):**
  $$\begin{bmatrix} 1 & 266.0 & 228.0 \\ 1 & 380.0 & 228.0 \\ 1 & 380.0 & 228.0 \\ 1 & 266.0 & 228.0 \\ 1 & 475.0 & 228.0 \\ 1 & 198.6 & 192.0 \end{bmatrix}, \quad y = \begin{bmatrix} 47.0298 \\ 43.6983 \\ 36.4478 \\ 45.8543 \\ 39.2898 \\ 38.0742 \end{bmatrix}$$

* **Normal Equation Matrices:**
  $$X^T X = \begin{bmatrix} 6.0000 & 1.9656 \times 10^3 & 1.3320 \times 10^3 \\ 1.9656 \times 10^3 & 6.9538 \times 10^5 & 4.4101 \times 10^5 \\ 1.3320 \times 10^3 & 4.4101 \times 10^5 & 2.9678 \times 10^5 \end{bmatrix}, \quad X^T y = \begin{bmatrix} 250.3942 \\ 8.1387 \times 10^4 \\ 5.5719 \times 10^4 \end{bmatrix}$$

* **Calculated Coefficients:**
  $$\beta_0 = -9.4096, \quad \beta_1 (\text{Cement}) = -0.0384, \quad \beta_2 (\text{Water}) = +0.2870$$

* **Sample Prediction & Residual (Observation 5):**
  $$\hat{y}_0 = -9.4096 - 0.0384(266.0) + 0.2870(228.0) = 45.8195 \text{ MPa}$$
  $$e_0 = y_0 - \hat{y}_0 = 47.0298 - 45.8195 = 1.2104 \text{ MPa}$$

* **Verification:** Scikit-Learn's `LinearRegression().fit()` produces identical parameters: $\beta_0 = -9.4096, \beta_1 = -0.0384, \beta_2 = +0.2870$.

---

## Model Evaluation and Performance

Model performance was evaluated on the **201 unseen test samples** (80/20 split, `random_state=42`) using five metrics:
- **Mean Absolute Error (MAE):** $\frac{1}{n} \sum |y_i - \hat{y}_i|$
- **Mean Squared Error (MSE):** $\frac{1}{n} \sum (y_i - \hat{y}_i)^2$
- **Root Mean Squared Error (RMSE):** $\sqrt{\text{MSE}}$
- **Coefficient of Determination ($R^2$):** $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$
- **Adjusted $R^2$ ($R^2_{\text{adj}}$):** $1 - \left[ \frac{(1 - R^2)(n - 1)}{n - p - 1} \right]$

### Evaluation Comparison Table

| Metric | Naive Mean Baseline ($\bar{y}_{\text{train}}$) | Single Linear Regression (`Cement`) | Multiple Linear Regression (All 8 Features) | Performance Improvement (vs. Single LR) |
| :--- | :---: | :---: | :---: | :---: |
| **MAE** | $13.9723 \text{ MPa}$ | $12.3955 \text{ MPa}$ | **$8.8960 \text{ MPa}$** | **-28.2%** error reduction |
| **MSE** | $299.1696$ | $236.4841$ | **$125.2653$** | **-47.0%** error reduction |
| **RMSE** | $17.2965 \text{ MPa}$ | $15.3780 \text{ MPa}$ | **$11.1922 \text{ MPa}$** | **-27.2%** error reduction |
| **$R^2$** | $-0.002821$ | $0.207302$ | **$0.580109$** | **+179.8%** explained variance |
| **Adjusted $R^2$** | $-0.002821$ | $0.203319$ | **$0.562613$** | **+176.7%** adjusted variance |

> **Note on Mean-Prediction Baseline:** When a `DummyRegressor` predicting the training mean ($\bar{y}_{\text{train}} = 35.0668 \text{ MPa}$) is evaluated on unseen test data ($\bar{y}_{\text{test}} = 35.9841 \text{ MPa}$), $R^2$ is slightly negative ($-0.002821$) rather than zero, because the test mean deviates from the training mean. For $p=0$, Adjusted $R^2 \equiv R^2$.

### Full Multiple Linear Regression Coefficients (Section 10)

$$\hat{y} = -23.2328 + 0.1166(\text{Cement}) + 0.0980(\text{Slag}) + 0.0846(\text{FlyAsh}) - 0.1317(\text{Water}) + 0.3301(\text{Superplasticizer}) + 0.0155(\text{CoarseAgg}) + 0.0206(\text{FineAgg}) + 0.1106(\text{Age})$$

*All coefficients are interpreted ceteris paribus (holding all other variables constant):*
- **`Cement` (+0.1166):** Each $+10 \text{ kg/m}^3$ of cement increases compressive strength by $+1.17 \text{ MPa}$.
- **`Slag` (+0.0980):** Supplementary binder contributes $+0.98 \text{ MPa}$ per $+10 \text{ kg/m}^3$.
- **`FlyAsh` (+0.0846):** Pozzolan contributes $+0.85 \text{ MPa}$ per $+10 \text{ kg/m}^3$.
- **`Water` (-0.1317):** Reduces strength by $-1.32 \text{ MPa}$ per $+10 \text{ kg/m}^3$, physically obeying Abrams' Law.
- **`Superplasticizer` (+0.3301):** Highest per-unit impact, reflecting high water-reduction efficiency.
- **`Age` (+0.1106):** Each additional curing day contributes $+0.11 \text{ MPa}$.

---

## Regression Diagnostics and Assumption Assessment

All five core assumptions of the Classical Linear Regression Model (CLRM) were systematically evaluated:

1. **Linearity of Parameters & Features:** Confirmed via bivariate scatter plots and Actual vs. Predicted alignment along the $45^\circ$ reference line ($R^2 = 0.58$).
2. **Independence of Errors (No Autocorrelation):** Satisfied; experimental observations represent independent cross-sectional batch trials partitioned via randomized 80/20 splitting.
3. **Homoscedasticity (Constant Error Variance):** Evaluated via the Residuals vs. Fitted scatter plot; errors remain uniformly dispersed across the $15 \text{ to } 55 \text{ MPa}$ range.
4. **Normality of Residuals:** Evaluated via the error distribution histogram and Kernel Density Estimate (KDE); residuals follow an approximately symmetric Gaussian bell curve centered at zero.
5. **Absence of Severe Multicollinearity:** Confirmed through Variance Inflation Factor diagnostics (all values $< 10.0$).

---

## Multicollinearity Analysis (VIF)

The Variance Inflation Factor quantifies the inflation of parameter variance due to linear collinearity: $\text{VIF}_j = \frac{1}{1 - R_j^2}$.

| Feature | Variance Inflation Factor (VIF) | Diagnostic Assessment |
| :--- | :---: | :--- |
| **`Cement`** | 7.4486 | Moderate collinearity ($< 10.0$) |
| **`Slag`** | 7.2628 | Moderate collinearity ($< 10.0$) |
| **`FineAggregate`** | 6.9726 | Moderate collinearity ($< 10.0$) |
| **`Water`** | 6.8392 | Moderate collinearity ($< 10.0$) |
| **`FlyAsh`** | 6.0854 | Moderate collinearity ($< 10.0$) |
| **`CoarseAggregate`** | 4.9576 | Low to moderate collinearity |
| **`Superplasticizer`** | 2.8687 | Low collinearity |
| **`Age`** | 1.1201 | Minimal collinearity |

**Engineering Assessment:** All VIF values remain strictly below the standard threshold of $10.0$. Moderate values for cement, slag, water, and aggregates arise naturally from the **unit-volume constraint** in concrete mix design (all components must sum to approximately $1.0 \text{ m}^3$). Retaining all 8 predictors preserves physical interpretability without destabilizing coefficient signs.

---

## New Mix Design Prediction

The trained model was applied to predict the compressive strength of an average industrial concrete mix design cured for 28 days:

* **Input Mix Proportions:**
  * `Cement`: $278.63 \text{ kg/m}^3$
  * `Slag`: $72.04 \text{ kg/m}^3$
  * `FlyAsh`: $55.54 \text{ kg/m}^3$
  * `Water`: $182.07 \text{ kg/m}^3$
  * `Superplasticizer`: $6.03 \text{ kg/m}^3$
  * `CoarseAggregate`: $974.38 \text{ kg/m}^3$
  * `FineAggregate`: $772.69 \text{ kg/m}^3$
  * `Age`: $28.0 \text{ days}$
* **Predicted Compressive Strength:** **`33.1344 MPa`**
* **Practical Context:** Matches commercial **Grade C30/C35 structural concrete** (target $30 \text{ to } 35 \text{ MPa}$), standard for structural slabs, beams, and columns.

---

## How to Run the Project

### Running in Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. Upload [`activity1_multiple_linear_regression.ipynb`](activity1_multiple_linear_regression.ipynb).
3. Click **Runtime** $\to$ **Run all** (`Ctrl + F9`).
4. The notebook automatically:
   * Executes the first code cell (`%pip install -q "xlrd>=2.0.1"`) to guarantee `.xls` Excel engine compatibility.
   * Downloads `Concrete_Data.xls` directly from the UCI repository.
   * Runs all 18 sections sequentially and renders all charts inline.

### Local Installation with requirements.txt

Clone the repository and install dependencies in a virtual environment:

```bash
# 1. Clone repository
git clone https://github.com/ease-up/ACTIVITY-1-MULTIPLE-LINEAR-REGRESSION-WORKSHEET.git
cd ACTIVITY-1-MULTIPLE-LINEAR-REGRESSION-WORKSHEET

# 2. Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install required packages
pip install -r requirements.txt

# 4. Launch Jupyter Notebook or JupyterLab
jupyter notebook activity1_multiple_linear_regression.ipynb
```

### Dataset Download Mechanism

The notebook includes automated dataset downloading in Code Cell 2:
```python
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
urllib.request.urlretrieve(url, "Concrete_Data.xls")
df = pd.read_excel("Concrete_Data.xls")
```
* **No manual download required:** Running the notebook fetches the dataset dynamically from the official UCI server if not already present locally.
* The local dataset file is tracked in `.gitignore` to prevent committing binary data files to the GitHub repository.

---

## Repository File Structure

```text
ACTIVITY-1-MULTIPLE-LINEAR-REGRESSION-WORKSHEET/
├── .gitignore                                  # Git exclusion rules (ignores dataset, backups, cache)
├── README.md                                   # Project documentation and engineering report
├── requirements.txt                            # Python package dependencies
└── activity1_multiple_linear_regression.ipynb  # Primary submission notebook (18 structured sections)
```

*(Note: `1st_activity.ipynb`, `.backup/`, and `Concrete_Data.xls` remain preserved on the local machine and are excluded from GitHub tracking via `.gitignore`.)*

---

## Engineering Decision Log and Limitations

### Professional Decision Log

| Stage | Decision Taken | Considered Alternative | Engineering & Statistical Justification |
| :--- | :--- | :--- | :--- |
| **Dataset Selection** | UCI Concrete Compressive Strength dataset (I-Cheng Yeh). | Synthetic data or custom small-batch test. | Real-world benchmark with 1,030 experimental formulations established in scientific literature. |
| **Data Cleaning** | Dropped 25 duplicate rows (retained 1,005 unique samples). | Retaining all duplicates. | Duplicate removal prevents artificial variance compression and eliminates data leakage across the train-test split. |
| **Manual Calculation Scope** | Normal equation on 6 samples using `Cement` and `Water`. | Manual calculation on entire dataset or 1 feature. | Contrasting one positive binder and one negative solvent clearly demonstrates multivariate matrix inversion without excessive tabular overhead. |
| **Baseline Benchmark** | Single Linear Regression on `Cement` ($X_1$) alongside naive mean baseline. | Purely reporting Multiple Linear Regression without comparison. | Establishes an empirical floor ($R^2 = 0.2073$) to quantify the actual value added by multivariate modeling ($R^2 = 0.5801$). |
| **Multicollinearity Handling** | Retained all 8 predictors despite moderate VIF values ($5 \text{ to } 7.5$). | Dropping features (e.g., Slag or Water) or applying PCA. | All VIFs are $< 10.0$. Dropping constituents breaks the volumetric balance and impairs physical interpretability. |
| **Evaluation Metrics** | Evaluated MAE, MSE, RMSE, $R^2$, and Adjusted $R^2$. | Reporting $R^2$ alone. | MAE provides an intuitive error metric in MPa, RMSE penalizes large errors, and Adjusted $R^2$ guards against model overparameterization. |

### Limitations & Future Enhancements
* **Curing Asymptote:** Hydration gain is rapid initially (1–28 days) and plateaus asymptotically toward 90–365 days. A linear model slightly under-predicts ultra-high-strength mixes ($>65 \text{ MPa}$).
* **Chemical Interactions:** Second-order interactions between superplasticizer and cement hydration are non-linear.
* **Future Work:** Incorporating logarithmic curing transformations ($\ln(\text{Age})$), water-binder interaction ratios, and non-linear ensemble models (Random Forest, Gradient Boosting) for enhanced accuracy.

---

## License

This project is submitted for college academic evaluation (CIE 2 Activity 1) and is released under the [MIT License](LICENSE).
