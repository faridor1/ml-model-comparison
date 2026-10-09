# Machine Learning Model Comparison: Linear Regression, XGBoost & Neural Networks 

A comparative machine learning project evaluating linear regression, gradient-boosted trees, and neural networks for a supervised tabular regression task. The workflow covers exploratory analysis, preprocessing, hyperparameter tuning, validation, and automated model selection for test-set predictions.

## Project overview

The goal is to compare four machine learning approaches on the same validation data and select the model with the lowest **root mean squared error (RMSE)** in the target's original units.

The dataset was supplied for coursework. Its original source, feature definitions, and target units have not been independently verified; the project therefore emphasizes the modeling workflow rather than domain-specific conclusions.

## Models compared

1. **Linear regression** — interpretable baseline.
2. **XGBoost regression** — gradient-boosted decision trees, tuned with five-fold cross-validation and grid search.
3. **Basic neural network** — feed-forward regression network.
4. **Tuned neural network** — Keras Tuner random search over architecture and regularization choices, with early stopping.

## Methodology

- Inspect feature distributions, missing values, correlations, and potential outliers.
- Split labeled observations into training and held-out validation subsets (80/20, random seed 42).
- Fit robust scaling parameters on training data only; apply those parameters to validation and test data.
- Train XGBoost on unscaled tabular features and tune its hyperparameters with cross-validation within the training subset.
- Reserve a separate tuning subset from the neural network's training data to avoid tuning directly against the final comparison set.
- Evaluate all four models on the same held-out validation set using **RMSE in original target units**.
- Automatically select the lowest-RMSE model and write predictions for the unlabeled test set to `Test_Predictions.csv`.

## Results

| Model | Validation RMSE (original units) |
|---|---:|
| **XGBoost** | **16.9924** |
| Tuned neural network | 19.3154 |
| Linear regression | 39.6243 |
| Basic neural network | 62.9284 |

**XGBoost achieved the lowest validation RMSE** and was selected for test-set prediction. These are validation results, not independently measured test-set errors, because test labels were not available in the notebook.

## Repository structure

```text
ml-model-comparison/
├── README.md
├── ml_model_comparison.ipynb
├── Test_Predictions.csv
├── requirements.txt
├── .gitignore
└── data/
    ├── README.md
    ├── train_X.csv
    ├── train_y.csv
    └── test_X.csv
```

## Setup and execution

1. Use a Python environment with the dependencies in `requirements.txt`.
2. Place the three CSV files in `data/` as shown above.
3. Open `ml_model_comparison.ipynb` in Jupyter or VS Code and run the cells in order.
4. The notebook saves `Test_Predictions.csv` in the repository root after model selection.

```bash
python -m pip install -r requirements.txt
```

Neural network tuning can take time and may generate local Keras Tuner files, which are excluded from version control.

## Limitations and next steps

- The notebook reports performance on one held-out validation split; results may vary with the split and random initialization.
- The supplied test set has no target labels in the notebook, so its prediction accuracy cannot be assessed here.
- Dataset provenance and the meaning of individual variables should be documented if the original course materials become available.
- A future extension could use repeated cross-validation, additional baselines, and feature-attribution analysis.

## Data provenance

Course-provided tabular dataset. See `data/README.md` for expected files and sharing considerations.
