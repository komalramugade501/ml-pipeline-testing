
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def process_data(data):
    # Check whether the dataset is empty
    if data.empty:
        raise ValueError("Dataset is empty")

    # Check whether the target column exists
    if "target" not in data.columns:
        raise ValueError("Missing target column")

    result = data.copy()

    # Identify input features
    feature_columns = [
        column for column in result.columns
        if column != "target"
    ]

    if not feature_columns:
        raise ValueError("No feature columns found")

    # Ensure all input features are numeric
    for column in feature_columns:
        if not pd.api.types.is_numeric_dtype(result[column]):
            raise TypeError(
                f"Feature '{column}' must be numeric"
            )

    # Replace infinite values with missing values
    result[feature_columns] = result[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    # Handle missing values using the median
    for column in feature_columns:
        if result[column].isna().all():
            raise ValueError(
                f"Feature '{column}' contains only missing values"
            )

        result[column] = result[column].fillna(
            result[column].median()
        )

    # Reject rows with missing target labels
    if result["target"].isna().any():
        raise ValueError("Target contains missing values")

    return result


def train_model():
    # Create the preprocessing and classification pipeline
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    return model


def validate_predictions(y_true, predictions):
    # Check prediction count
    if len(y_true) != len(predictions):
        raise ValueError("Prediction count does not match")

    # Check for missing predictions
    if pd.isna(predictions).any():
        raise ValueError("Predictions contain missing values")

    # Check that predicted labels are valid Iris classes
    allowed_classes = {0, 1, 2}

    if not set(predictions).issubset(allowed_classes):
        raise ValueError("Predictions contain unknown class")

    return True


def evaluate_model(y_true, predictions):
    # Calculate model accuracy
    return accuracy_score(y_true, predictions)
