"""Load the saved classifier and predict one held-out dataset example."""
from pathlib import Path
import argparse

import joblib
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path,
        default=Path(__file__).resolve().parent / "models" / "logistic_regression.joblib",
        help="Path to the model file created by the notebook.",
    )
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error("Model file not found. Run the notebook's final training cell first.")

    # Load only the trusted model file created by this project.
    bundle = joblib.load(args.model)
    if bundle["sklearn_version"] != sklearn.__version__:
        parser.error(
            f"Model requires scikit-learn {bundle['sklearn_version']}; "
            f"installed version is {sklearn.__version__}. "
            "Recreate the model with the notebook's final cell in this environment."
        )

    # Recreate the test split and select one case that was not used for training.
    data = load_breast_cancer(as_frame=True)
    target = (data.target == 0).astype(int)
    _, X_test, _, y_test = train_test_split(
        data.data, target, test_size=0.20, stratify=target, random_state=42
    )
    sample = X_test.iloc[[0]].loc[:, bundle["feature_names"]]

    # Reuse the saved preprocessing and apply the saved threshold explicitly.
    model = bundle["pipeline"]
    malignant_column = list(model.classes_).index(1)
    probability = float(model.predict_proba(sample)[0, malignant_column])
    prediction = int(probability >= bundle["threshold"])

    print(f"Example dataset row: {sample.index[0]}")
    print(f"Malignant probability: {probability:.2%}")
    print(f"Threshold: {bundle['threshold']:.2f}")
    print(f"Predicted class: {bundle['labels'][prediction]}")
    print(f"Actual class: {bundle['labels'][int(y_test.iloc[0])]}")


if __name__ == "__main__":
    main()
