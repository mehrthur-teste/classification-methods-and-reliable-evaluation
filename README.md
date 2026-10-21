# Classification Methods and Reliable Evaluation

## What does In_Class_Activity.ipynb do?

The file `MCAI500_M3_In_Class_Activity.ipynb` is a practical Machine Learning exercise for classifying tumors as benign or malignant and comparing prediction quality.

The notebook:

1. **Loads the data** describing tumor characteristics and sets malignant to `1` and benign to `0`.
2. **Splits the data** into 80% for training and 20% for testing, preserving similar class proportions.
3. **Trains three models:** Logistic Regression, KNN, and Random Forest.
4. **Compares their results** using accuracy, precision, recall, and F1.

It then includes activities for experimenting with model hyperparameters, interpreting a confusion matrix, performing cross-validation, and testing different probability thresholds for classifying a case as malignant.

Currently, Tasks 1 through 4 have their code filled in. Tasks 5 through 8 still need to be completed and include commented examples.

The goal is to understand which model detects malignant cases most effectively and how configuration choices affect errors, rather than looking only at the overall percentage of correct predictions.

## Explanation of the code lines

### Import the function that loads the data

```python
from sklearn.datasets import load_breast_cancer
```

Imports the `load_breast_cancer` function from the `datasets` module of the scikit-learn (`sklearn`) library. This function loads a dataset of breast tumor characteristics used in this exercise to predict whether a tumor is benign or malignant.

This line only imports the function. The data is loaded when the function is called, for example:

```python
dataset = load_breast_cancer(as_frame=True)
```

The `as_frame=True` argument provides the data in pandas structures, making it easier to inspect and manipulate.

### Define a seed for reproducible results

```python
RANDOM_STATE = 42
```

Creates a variable with the value `42`, used as a seed for operations involving randomness. For example, passing `random_state=RANDOM_STATE` to the data-splitting function produces the same split when execution is repeated with the same data and conditions.

The number `42` is a conventional choice; another integer could be used. Keeping the same value helps reproduce and compare results. This assignment alone does not configure randomness across all libraries: the variable must be passed to functions that accept it.

### Adjust how tables are displayed

```python
pd.set_option("display.max_columns", 12)
```

Configures pandas, imported as `pd`, to display up to 12 columns when showing a table (`DataFrame`). When there are more columns, pandas may shorten the display using `...`.

This option only affects the display. It does not remove columns or modify the data.

### Configure the chart style

```python
sns.set_theme(style="whitegrid")
```

Configures the chart theme using seaborn, imported as `sns`. The `whitegrid` style uses a white background with grid lines, making values easier to read and compare.

This line changes the appearance of charts. It does not affect the data, model training, or predictions.

## Save and reuse the selected model

1. Install the dependencies: `python -m pip install -r requirements.txt`.
2. Run the notebook's final **Train and save the selected model** cell from the project folder. It trains on the original training split and saves `models/logistic_regression.joblib`.
3. Run `python predict_example.py` to load the saved pipeline and classify one held-out example without retraining. The script also works from another directory because its default model path is relative to the script.

The saved bundle includes preprocessing, the 30 feature names, class labels, the scikit-learn version, and threshold **0.30**. The example uses the threshold explicitly rather than the default `predict()` threshold of 0.50. Recreate the model using the final notebook cell if your scikit-learn version differs from the saved version. Only load trusted joblib files.

The threshold is an exploratory choice from this assignment, rather than an independently validated result.
