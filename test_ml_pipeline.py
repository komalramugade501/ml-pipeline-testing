
import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ml_pipeline import (
    process_data,
    train_model,
    validate_predictions,
    evaluate_model
)


@pytest.fixture
def sample_data():
    iris = load_iris(as_frame=True)
    return iris.frame.copy()


def test_process_valid_data(sample_data):
    result = process_data(sample_data)

    assert result.shape == sample_data.shape
    assert result.isnull().sum().sum() == 0


def test_process_missing_feature_values(sample_data):
    sample_data.loc[0, "sepal length (cm)"] = np.nan

    result = process_data(sample_data)

    assert result["sepal length (cm)"].isnull().sum() == 0


def test_missing_target_column(sample_data):
    sample_data = sample_data.drop(columns=["target"])

    with pytest.raises(ValueError, match="Missing target column"):
        process_data(sample_data)


def test_non_numeric_feature(sample_data):
    sample_data["sepal length (cm)"] = "invalid"

    with pytest.raises(TypeError, match="must be numeric"):
        process_data(sample_data)


def test_empty_dataset():
    empty_data = pd.DataFrame(
        columns=["sepal length (cm)", "target"]
    )

    with pytest.raises(ValueError, match="Dataset is empty"):
        process_data(empty_data)


def test_training_pipeline(sample_data):
    data = process_data(sample_data)

    X = data.drop(columns=["target"])
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)


def test_prediction_count_mismatch():
    with pytest.raises(
        ValueError, match="Prediction count does not match"
    ):
        validate_predictions([0, 1, 2], [0, 1])


def test_unknown_prediction_class():
    with pytest.raises(
        ValueError, match="Predictions contain unknown class"
    ):
        validate_predictions([0, 1], [0, 3])


def test_evaluation_metric(sample_data):
    y_true = [0, 1, 2]
    predictions = [0, 1, 2]

    accuracy = evaluate_model(y_true, predictions)

    assert accuracy == 1.0
    assert accuracy == accuracy_score(y_true, predictions)
