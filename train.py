from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from utils import (
    ANOMALY_MODEL_PATH,
    CLASSIFICATION_MODEL_PATH,
    METRICS_PATH,
    PLOTS_DIR,
    REGRESSION_FEATURES,
    REGRESSION_MODEL_PATH,
    add_date_features,
    clean_expense_data,
    ensure_directories,
    load_or_create_dataset,
    prepare_anomaly_features,
    save_pickle,
)


def train_regression_model(dataframe):
    """Train a regression model to predict expense amount."""
    working = add_date_features(clean_expense_data(dataframe)).sort_values("date").reset_index(drop=True)
    model_frame = working[REGRESSION_FEATURES + ["amount"]].copy()

    split_index = max(1, int(len(model_frame) * 0.8))
    train_frame = model_frame.iloc[:split_index]
    test_frame = model_frame.iloc[split_index:]

    if test_frame.empty:
        train_frame, test_frame = train_test_split(model_frame, test_size=0.2, random_state=42)

    x_train = train_frame[REGRESSION_FEATURES]
    y_train = train_frame["amount"]
    x_test = test_frame[REGRESSION_FEATURES]
    y_test = test_frame["amount"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), ["category", "payment_method"]),
            ("numerical", "passthrough", ["month", "day", "day_of_week", "week_of_year", "is_weekend"]),
        ]
    )

    regression_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )

    regression_pipeline.fit(x_train, y_train)
    predictions = regression_pipeline.predict(x_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    metrics = {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "actual": y_test.tolist(),
        "predicted": predictions.tolist(),
    }
    return regression_pipeline, metrics


def train_category_model(dataframe):
    """Train an NLP classifier to predict the expense category from text."""
    working = clean_expense_data(dataframe)
    x_train, x_test, y_train, y_test = train_test_split(
        working["description"],
        working["category"],
        test_size=0.2,
        random_state=42,
        stratify=working["category"],
    )

    classification_pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    classification_pipeline.fit(x_train, y_train)
    predictions = classification_pipeline.predict(x_test)

    labels = sorted(working["category"].unique())
    confusion = confusion_matrix(y_test, predictions, labels=labels)
    accuracy = accuracy_score(y_test, predictions)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "labels": labels,
        "confusion_matrix": confusion.tolist(),
    }
    return classification_pipeline, metrics


def train_anomaly_detector(dataframe):
    """Train Isolation Forest to identify unusual spending behaviour."""
    features, feature_columns = prepare_anomaly_features(dataframe)
    model = IsolationForest(contamination=0.03, random_state=42)
    model.fit(features)

    predictions = model.predict(features)
    anomaly_count = int((predictions == -1).sum())

    artifact = {"model": model, "feature_columns": feature_columns}
    metrics = {"anomaly_count": anomaly_count}
    return artifact, metrics


def save_visualizations(dataframe, regression_metrics: dict, classification_metrics: dict) -> None:
    """Create and save the requested project charts."""
    working = add_date_features(clean_expense_data(dataframe))

    monthly_spend = working.groupby("year_month")["amount"].sum().reset_index()
    category_spend = working.groupby("category")["amount"].sum().sort_values(ascending=False)

    plt.figure(figsize=(12, 5))
    sns.lineplot(data=monthly_spend, x="year_month", y="amount", marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.title("Monthly Expense Trends")
    plt.xlabel("Month")
    plt.ylabel("Total Spend")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "monthly_expense_trends.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(
        x=category_spend.index,
        y=category_spend.values,
        hue=category_spend.index,
        dodge=False,
        legend=False,
        palette="crest",
    )
    plt.xticks(rotation=30, ha="right")
    plt.title("Category-wise Spending")
    plt.xlabel("Category")
    plt.ylabel("Total Spend")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "category_spending.png")
    plt.close()

    actual = regression_metrics["actual"][:60]
    predicted = regression_metrics["predicted"][:60]
    sample_index = list(range(1, len(actual) + 1))

    plt.figure(figsize=(12, 5))
    plt.plot(sample_index, actual, label="Actual", linewidth=2)
    plt.plot(sample_index, predicted, label="Predicted", linewidth=2)
    plt.title("Prediction vs Actual Expense Amount")
    plt.xlabel("Test Sample")
    plt.ylabel("Amount")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "prediction_vs_actual.png")
    plt.close()

    labels = classification_metrics["labels"]
    confusion = np.array(classification_metrics["confusion_matrix"])

    plt.figure(figsize=(9, 7))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix for Category Prediction")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png")
    plt.close()


def run_training_pipeline(force_generate: bool = False) -> dict:
    """Run the full ML workflow from data loading to model saving."""
    ensure_directories()
    dataframe = load_or_create_dataset(force_generate=force_generate, num_rows=1500)

    regression_model, regression_metrics = train_regression_model(dataframe)
    classification_model, classification_metrics = train_category_model(dataframe)
    anomaly_artifact, anomaly_metrics = train_anomaly_detector(dataframe)

    save_pickle(regression_model, REGRESSION_MODEL_PATH)
    save_pickle(classification_model, CLASSIFICATION_MODEL_PATH)
    save_pickle(anomaly_artifact, ANOMALY_MODEL_PATH)

    metrics = {
        "regression": regression_metrics,
        "classification": classification_metrics,
        "anomaly": anomaly_metrics,
    }
    save_pickle(metrics, METRICS_PATH)
    save_visualizations(dataframe, regression_metrics, classification_metrics)

    return metrics


if __name__ == "__main__":
    metrics = run_training_pipeline()
    print("Training completed successfully.")
    print(f"Regression MAE: {metrics['regression']['mae']}")
    print(f"Regression RMSE: {metrics['regression']['rmse']}")
    print(f"Classification Accuracy: {metrics['classification']['accuracy']}")
    print(f"Detected anomalies in dataset: {metrics['anomaly']['anomaly_count']}")
    print(f"Saved models to: {Path('models').resolve()}")
