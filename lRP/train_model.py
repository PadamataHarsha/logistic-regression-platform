import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "advertising.csv"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
IMAGE_DIR = ROOT_DIR / "static" / "images"
FEATURE_COLUMNS = [
    "Daily Time Spent on Site",
    "Age",
    "Area Income",
    "Daily Internet Usage",
    "Male",
]
TARGET_COLUMN = "Clicked on Ad"
FEATURE_PRESENTATION = {
    "Daily Time Spent on Site": (
        "Daily time on site",
        "Use the value recorded for time spent on the site in the source dataset's units.",
    ),
    "Age": ("Age", "Enter the age value recorded in the dataset."),
    "Area Income": (
        "Area income",
        "Use the income value in the same units as the source dataset.",
    ),
    "Daily Internet Usage": (
        "Daily internet usage",
        "Use the usage value recorded in the source dataset's units.",
    ),
    "Male": ("Gender indicator", "The dataset encodes this field as 0 or 1."),
}


def _save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def train_model() -> dict:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [name for name in required_columns if name not in data.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if data[required_columns].isnull().any().any():
        raise ValueError("Model columns contain missing values; clean the CSV before training.")

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)
    class_values = [int(value) for value in pipeline.classes_]
    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=pipeline.classes_)
    labels = {
        str(value): (
            "Likely to Click" if value == 1 else "Unlikely to Click"
        )
        if TARGET_COLUMN == "Clicked on Ad"
        else str(value)
        for value in class_values
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "trained_model.joblib")
    _save_json(OUTPUT_DIR / "classification_report.json", report)
    pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "predicted": predictions,
            **{
                f"probability_class_{class_value}": probabilities[:, index]
                for index, class_value in enumerate(class_values)
            },
        },
        index=y_test.index,
    ).sort_index().to_csv(OUTPUT_DIR / "predictions.csv", index_label="dataset_row")

    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=[labels[str(value)] for value in class_values],
        yticklabels=[labels[str(value)] for value in class_values],
        ax=axis,
    )
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_title("Holdout confusion matrix")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    figure.savefig(IMAGE_DIR / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    target_counts = y.value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(7, 4))
    sns.barplot(
        x=[labels.get(str(value), str(value)) for value in target_counts.index],
        y=target_counts.values,
        hue=[labels.get(str(value), str(value)) for value in target_counts.index],
        palette=["#247b70", "#d28a46"][: len(target_counts)],
        legend=False,
        ax=axis,
    )
    axis.set_title("Target-class distribution")
    axis.set_xlabel("")
    axis.set_ylabel("Records")
    figure.tight_layout()
    figure.savefig(IMAGE_DIR / "target_distribution.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, column in zip(axes.flat, FEATURE_COLUMNS):
        sns.histplot(data=data, x=column, bins=24, kde=column != "Male", ax=axis)
        axis.set_title(column)
    axes.flat[-1].axis("off")
    figure.suptitle("Training dataset feature distributions")
    figure.tight_layout()
    figure.savefig(IMAGE_DIR / "feature_distributions.png", dpi=150)
    plt.close(figure)

    classifier = pipeline.named_steps["logistic_regression"]
    coefficients = {
        feature: float(coefficient)
        for feature, coefficient in zip(FEATURE_COLUMNS, classifier.coef_[0])
    }
    feature_details = []
    for name in FEATURE_COLUMNS:
        series = data[name]
        feature_details.append(
            {
                "name": name,
                "display_name": FEATURE_PRESENTATION[name][0],
                "description": FEATURE_PRESENTATION[name][1],
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "kind": "binary" if name == "Male" else "numeric",
                "coefficient": coefficients[name],
            }
        )
    metadata = {
        "model_name": "StandardScaler + Logistic Regression",
        "model_status": "trained",
        "last_trained": datetime.now(timezone.utc).isoformat(),
        "features": feature_details,
        "feature_names": FEATURE_COLUMNS,
        "feature_order": FEATURE_COLUMNS,
        "target_name": TARGET_COLUMN,
        "class_labels": labels,
        "metrics": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        },
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "class_values": class_values,
        "target_distribution": {
            labels.get(str(value), str(value)): int(count)
            for value, count in target_counts.items()
        },
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "dataset_records": int(len(data)),
        "dataset_columns": [str(name) for name in data.columns],
        "dataset_shape": [int(data.shape[0]), int(data.shape[1])],
        "missing_values": {str(name): int(count) for name, count in data.isna().sum().items()},
        "numeric_summary": json.loads(
            data.select_dtypes(include="number").describe().to_json()
        ),
    }
    _save_json(MODEL_DIR / "model_metadata.json", metadata)
    print(f"Trained {metadata['model_name']} using {len(X_train)} training records.")
    print(f"Holdout accuracy: {metadata['metrics']['accuracy']:.3f}")
    print(f"Saved model and evaluation outputs under {ROOT_DIR}")
    return metadata


if __name__ == "__main__":
    train_model()