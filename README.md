# House Prices — Advanced Regression Techniques

An end-to-end, notebook-based machine learning study for predicting residential sale prices using the **Ames Housing** dataset. The project covers data understanding, exploratory analysis, preprocessing, regression model selection, and evaluation on a held-out subset.

The emphasis is on **reproducible experiments, leakage-aware preprocessing, and interpreting model errors**, rather than simply reporting the lowest score.

## Problem and dataset

Residential sale prices depend on many interacting factors, including property size, construction quality, neighborhood, and amenities. This project investigates how well these characteristics can predict `SalePrice`.

- **Source:** [Ames Housing on OpenML (data ID 42165)](https://www.openml.org/d/42165)
- **Observations:** 1,460 houses
- **Input features:** 79 after dropping the record identifier `Id`
- **Target:** `SalePrice`
- **Split:** 80% training (1,168 rows) and 20% holdout (292 rows), using `random_state=42`
- **Evaluation:** shuffled 5-fold cross-validation on training data; root mean squared error (RMSE) on `log1p(SalePrice)`

The holdout is kept out of preprocessing fitting and training-time model selection. Because earlier development iterations also inspected this same holdout, its reported score should **not** be treated as performance on a fresh, external test dataset.

## Project structure

```text
.
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   └── 04_model_comparison.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

| Notebook | Purpose |
| --- | --- |
| [01 — Data Understanding](notebooks/01_data_understanding.ipynb) | Inspect types, target distribution, outliers, and missing-value patterns; establish the train/holdout split. |
| [02 — Exploratory Data Analysis](notebooks/02_eda.ipynb) | Explore numerical and categorical predictors, associations with sale price, and the log-transformed target. |
| [03 — Preprocessing](notebooks/03_preprocessing.ipynb) | Compare imputation and scaling strategies, construct scikit-learn pipelines, and examine Ridge regularization. |
| [04 — Model Comparison](notebooks/04_model_comparison.ipynb) | Tune and compare five model families, evaluate the selected model, and analyze residuals and feature importance. |

Each notebook loads the dataset from OpenML and reconstructs the same split, so the notebooks can be run separately. Run them in numbered order for the complete narrative.

## Approach

1. **Understand the data.** Inspect feature types, missingness, the right-skewed price distribution, and candidate outliers without deleting observations solely on the basis of an IQR rule.
2. **Explore training data.** Analyze relationships and try `log1p(SalePrice)` as the regression target. Avoid using holdout outcomes to drive exploratory feature decisions.
3. **Build preprocessing pipelines.** Drop `Id`; treat the numerically coded `MSSubClass` as categorical; impute other numerical columns with medians; encode categorical columns with `OneHotEncoder(handle_unknown="ignore")` and an explicit missing-value category. Evaluate neighborhood-based median imputation for `LotFrontage`.
4. **Avoid data leakage.** Fit imputers, encoders, and any numerical scaler within scikit-learn `Pipeline` / `ColumnTransformer` objects during each cross-validation training fold.
5. **Compare model families.** Evaluate Ridge, Lasso, ElasticNet, Random Forest, and Gradient Boosting using the same shuffled 5-fold CV configuration. Use scaling for regularized linear models; tree ensembles do not require it.
6. **Evaluate and interpret.** Select a configuration using training CV, then report the holdout score and investigate error patterns and feature importance without tuning further on the holdout.

## Results

The following results are from the saved outputs of `04_model_comparison.ipynb` after a clean restart and full execution.

| Model | Mean 5-fold CV RMSE ↓ |
| --- | ---: |
| **Gradient Boosting** | **0.126787** |
| ElasticNet | 0.138241 |
| Lasso | 0.138643 |
| Ridge | 0.139543 |
| Random Forest | 0.139974 |

**Selected configuration:** `GradientBoostingRegressor(learning_rate=0.1, n_estimators=300, max_depth=2, random_state=42)`

**Holdout RMSE:** **0.137880**, evaluated on 292 held-out houses.

> All RMSE values in this section are measured on `log1p(SalePrice)`, **not in US dollars**. Hyperparameters were chosen through repeated training-set CV comparisons, so the winning CV score may be optimistic. Holdout-based error analysis is descriptive, not a basis for additional tuning.

### Key observations

- Missing values are sometimes meaningful: for example, missing garage and basement attributes can correspond to houses without those structures.
- Neighborhood-informed `LotFrontage` imputation was tested, but did not improve the linear baseline in the preprocessing experiments.
- A solver-tolerance investigation showed that sparse ordinary least-squares CV scores were sensitive to numerical convergence settings; the final preprocessing comparison uses `LinearRegression(tol=1e-10)` consistently.
- The selected Gradient Boosting model tends to **underpredict higher-priced houses** in the holdout price-segment analysis.
- Feature-importance analyses highlight factors such as overall quality and living area. Such importance measures describe the fitted model; they do not establish causality.

## Reproduce the analysis

**Requirements:** Python 3.12 or newer, internet access to download the OpenML dataset on first use, and Jupyter or VS Code with the Jupyter extension.

```bash
# Clone the repository
git clone https://github.com/masoud-vosoughi/House-Prices--Advanced-Regression-Techniques.git
cd House-Prices--Advanced-Regression-Techniques

# Create and activate a virtual environment (macOS / Linux)
python3 -m venv .venv
source .venv/bin/activate

# Install the project dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Make the environment available as a notebook kernel
python -m ipykernel install --user --name house-prices-ml --display-name "Python (house-prices-ml)"
```

Open the `notebooks/` files in VS Code or Jupyter, select the installed kernel, and use **Restart & Run All**. All notebooks fetch the Ames data directly through `sklearn.datasets.fetch_openml(data_id=42165, as_frame=True)`; no local dataset file is required.

On Windows, activate the environment with `.venv\Scripts\activate` instead of the macOS/Linux command shown above.

## Tools

**Python**, **pandas**, **NumPy**, **scikit-learn**, **SciPy**, **Matplotlib**, **Jupyter notebooks**, and **Git**.

## Limitations and future work

- The dataset is relatively small, and scores are sensitive to cross-validation partitions and model selection.
- A single 80/20 holdout does not establish performance on new markets or different periods; moreover, the holdout was inspected during earlier development iterations.
- Model errors are not uniform across price ranges, particularly for high-value homes.
- Future experiments could use nested cross-validation, an untouched external test set, uncertainty estimates, and additional feature engineering. Any improvements should be selected without using existing holdout errors for tuning.

## Outputs

This repository contains **reproducible notebooks and their analysis outputs**. It does not currently provide a hosted web application or prediction API. Start with [the model-comparison notebook](notebooks/04_model_comparison.ipynb) for the model leaderboard, holdout metrics, error analysis, and feature-importance visualizations.
