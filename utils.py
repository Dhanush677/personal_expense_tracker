from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = DATA_DIR / "plots"
MODELS_DIR = BASE_DIR / "models"

DATA_PATH = DATA_DIR / "expenses.csv"
REGRESSION_MODEL_PATH = MODELS_DIR / "expense_regressor.pkl"
CLASSIFICATION_MODEL_PATH = MODELS_DIR / "category_classifier.pkl"
ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_detector.pkl"
METRICS_PATH = MODELS_DIR / "metrics.pkl"

REGRESSION_FEATURES = [
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    "is_weekend",
    "category",
    "payment_method",
]

CATEGORIES = [
    "Food",
    "Travel",
    "Bills",
    "Shopping",
    "Entertainment",
    "Health",
    "Groceries",
    "Education",
]

PAYMENT_METHODS = ["UPI", "Credit Card", "Debit Card", "Cash", "Net Banking"]

CATEGORY_PROBABILITIES = {
    "Food": 0.21,
    "Travel": 0.10,
    "Bills": 0.14,
    "Shopping": 0.13,
    "Entertainment": 0.11,
    "Health": 0.08,
    "Groceries": 0.17,
    "Education": 0.06,
}

CATEGORY_TEMPLATES = {
    "Food": {
        "mean": 340,
        "std": 120,
        "descriptions": [
            "Swiggy order",
            "Zomato dinner",
            "Cafe breakfast",
            "Office lunch",
            "Pizza takeaway",
            "Family dinner",
            "Street food snack",
            "Restaurant bill",
        ],
        "payment_weights": [0.55, 0.20, 0.12, 0.08, 0.05],
    },
    "Travel": {
        "mean": 900,
        "std": 450,
        "descriptions": [
            "Uber ride",
            "Metro recharge",
            "Flight booking",
            "Bus ticket",
            "Train booking",
            "Cab to office",
            "Fuel refill",
            "Parking charge",
        ],
        "payment_weights": [0.36, 0.24, 0.18, 0.10, 0.12],
    },
    "Bills": {
        "mean": 2200,
        "std": 900,
        "descriptions": [
            "Electricity bill",
            "Water bill",
            "Mobile recharge",
            "Internet payment",
            "Gas bill",
            "Rent payment",
            "Insurance premium",
            "Subscription renewal",
        ],
        "payment_weights": [0.22, 0.12, 0.18, 0.04, 0.44],
    },
    "Shopping": {
        "mean": 1800,
        "std": 1100,
        "descriptions": [
            "Amazon purchase",
            "Flipkart order",
            "Clothing store",
            "Electronics accessory",
            "Home decor item",
            "Gift purchase",
            "Shoe shopping",
            "Online shopping cart",
        ],
        "payment_weights": [0.24, 0.34, 0.18, 0.08, 0.16],
    },
    "Entertainment": {
        "mean": 700,
        "std": 300,
        "descriptions": [
            "Movie tickets",
            "OTT subscription",
            "Concert pass",
            "Gaming recharge",
            "Weekend outing",
            "Theme park entry",
            "Streaming service payment",
            "Bowling with friends",
        ],
        "payment_weights": [0.30, 0.26, 0.18, 0.12, 0.14],
    },
    "Health": {
        "mean": 1200,
        "std": 650,
        "descriptions": [
            "Pharmacy purchase",
            "Doctor consultation",
            "Dental checkup",
            "Medical test",
            "Gym membership",
            "Health supplement",
            "Hospital visit",
            "Eye checkup",
        ],
        "payment_weights": [0.28, 0.26, 0.18, 0.10, 0.18],
    },
    "Groceries": {
        "mean": 1400,
        "std": 500,
        "descriptions": [
            "BigBasket order",
            "Local grocery shop",
            "Supermarket shopping",
            "Vegetable market",
            "Dairy products purchase",
            "Household essentials",
            "Monthly groceries",
            "Fresh produce order",
        ],
        "payment_weights": [0.40, 0.16, 0.18, 0.18, 0.08],
    },
    "Education": {
        "mean": 2600,
        "std": 1300,
        "descriptions": [
            "Online course fee",
            "Book purchase",
            "Exam registration",
            "Tuition payment",
            "Certification fee",
            "Workshop ticket",
            "Study material order",
            "Coaching class fee",
        ],
        "payment_weights": [0.22, 0.24, 0.18, 0.05, 0.31],
    },
}


def ensure_directories() -> None:
    """Create the folders used by the project if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def save_pickle(obj: object, path: Path) -> None:
    """Save any Python object as a pickle file."""
    with path.open("wb") as file:
        pickle.dump(obj, file)


def load_pickle(path: Path) -> object:
    """Load a pickle file and return its content."""
    with path.open("rb") as file:
        return pickle.load(file)


def generate_synthetic_expense_data(num_rows: int = 1500, seed: int = 42) -> pd.DataFrame:
    """
    Create a realistic synthetic expense dataset.

    The data includes repeating monthly patterns, category-specific amounts,
    different payment preferences, a few missing values, and some outliers.
    """
    ensure_directories()

    rng = np.random.default_rng(seed)
    available_dates = pd.date_range(start="2023-01-01", end="2025-12-31", freq="D")

    chosen_categories = rng.choice(
        CATEGORIES,
        size=num_rows,
        p=[CATEGORY_PROBABILITIES[category] for category in CATEGORIES],
    )
    chosen_dates = rng.choice(available_dates, size=num_rows, replace=True)

    records: List[Dict[str, object]] = []

    for date_value, category in zip(chosen_dates, chosen_categories):
        template = CATEGORY_TEMPLATES[category]
        payment_method = rng.choice(PAYMENT_METHODS, p=template["payment_weights"])
        description = rng.choice(template["descriptions"])

        mean_amount = float(template["mean"])
        std_amount = float(template["std"])

        date_value = pd.Timestamp(date_value)
        weekend_multiplier = 1.15 if date_value.dayofweek >= 5 else 1.0
        long_term_trend = 1 + ((date_value.year - 2023) * 0.06)
        month_multiplier = 1.0

        if category == "Travel" and date_value.month in [4, 5, 6, 11, 12]:
            month_multiplier *= 1.30
        if category == "Shopping" and date_value.month in [10, 11, 12]:
            month_multiplier *= 1.25
        if category == "Bills" and date_value.day <= 7:
            month_multiplier *= 1.10
        if category in ["Food", "Entertainment"] and date_value.dayofweek >= 4:
            month_multiplier *= 1.18
        if category == "Education" and date_value.month in [6, 7, 8]:
            month_multiplier *= 1.20

        amount = rng.normal(mean_amount, std_amount) * weekend_multiplier * month_multiplier * long_term_trend
        amount = round(max(40, amount), 2)

        records.append(
            {
                "date": date_value.date(),
                "amount": amount,
                "category": category,
                "description": description,
                "payment_method": payment_method,
            }
        )

    dataframe = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

    anomaly_count = max(20, int(num_rows * 0.03))
    anomaly_indices = rng.choice(dataframe.index, size=anomaly_count, replace=False)
    anomaly_multipliers = rng.uniform(2.5, 6.0, size=anomaly_count)
    dataframe.loc[anomaly_indices, "amount"] = (
        dataframe.loc[anomaly_indices, "amount"].to_numpy() * anomaly_multipliers
    ).round(2)

    for column_name, missing_ratio in {
        "description": 0.015,
        "payment_method": 0.01,
        "category": 0.01,
        "amount": 0.008,
    }.items():
        missing_indices = rng.choice(
            dataframe.index,
            size=max(1, int(num_rows * missing_ratio)),
            replace=False,
        )
        dataframe.loc[missing_indices, column_name] = np.nan

    return dataframe


def clean_expense_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and normalize column types."""
    cleaned = dataframe.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned["amount"] = pd.to_numeric(cleaned["amount"], errors="coerce")

    cleaned["amount"] = cleaned["amount"].fillna(cleaned["amount"].median())
    cleaned["description"] = cleaned["description"].fillna("miscellaneous expense")
    cleaned["category"] = cleaned["category"].fillna(cleaned["category"].mode().iloc[0])
    cleaned["payment_method"] = cleaned["payment_method"].fillna(cleaned["payment_method"].mode().iloc[0])
    cleaned = cleaned.dropna(subset=["date"]).reset_index(drop=True)
    return cleaned


def add_date_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Extract useful ML features from the date column."""
    featured = dataframe.copy()
    featured["date"] = pd.to_datetime(featured["date"])
    featured["month"] = featured["date"].dt.month
    featured["day"] = featured["date"].dt.day
    featured["day_of_week"] = featured["date"].dt.dayofweek
    featured["week_of_year"] = featured["date"].dt.isocalendar().week.astype(int)
    featured["year"] = featured["date"].dt.year
    featured["year_month"] = featured["date"].dt.to_period("M").astype(str)
    featured["is_weekend"] = (featured["day_of_week"] >= 5).astype(int)
    return featured


def load_or_create_dataset(force_generate: bool = False, num_rows: int = 1500) -> pd.DataFrame:
    """
    Load the CSV dataset from disk.
    If it does not exist, generate a fresh synthetic dataset and save it.
    """
    ensure_directories()

    if force_generate or not DATA_PATH.exists():
        dataframe = generate_synthetic_expense_data(num_rows=num_rows)
        dataframe.to_csv(DATA_PATH, index=False)
        return dataframe

    return pd.read_csv(DATA_PATH)


def prepare_anomaly_features(
    dataframe: pd.DataFrame,
    feature_columns: List[str] | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a numeric feature matrix for anomaly detection.

    Isolation Forest works only with numeric data, so categorical fields are
    one-hot encoded here.
    """
    working = add_date_features(clean_expense_data(dataframe))
    feature_frame = working[
        [
            "amount",
            "month",
            "day",
            "day_of_week",
            "week_of_year",
            "is_weekend",
            "category",
            "payment_method",
        ]
    ].copy()
    encoded = pd.get_dummies(feature_frame, columns=["category", "payment_method"], dtype=float)

    if feature_columns is not None:
        encoded = encoded.reindex(columns=feature_columns, fill_value=0.0)

    return encoded, list(encoded.columns)


def get_top_spending_categories(dataframe: pd.DataFrame, top_n: int = 3) -> pd.Series:
    """Return the categories with the highest total spending."""
    working = clean_expense_data(dataframe)
    return working.groupby("category")["amount"].sum().sort_values(ascending=False).head(top_n)


def get_budget_suggestion(dataframe: pd.DataFrame) -> float:
    """
    Suggest next month's budget based on recent monthly spending.

    A small 5% buffer is added so the number is practical for planning.
    """
    working = add_date_features(clean_expense_data(dataframe))
    monthly_totals = working.groupby("year_month")["amount"].sum().sort_index()
    recent_months = monthly_totals.tail(3)

    if recent_months.empty:
        return 0.0

    return round(recent_months.mean() * 1.05, 2)


def create_regression_input(expense_date: str, category: str, payment_method: str) -> pd.DataFrame:
    """Create a one-row DataFrame for future expense prediction."""
    input_frame = pd.DataFrame(
        [
            {
                "date": expense_date,
                "category": category,
                "payment_method": payment_method,
            }
        ]
    )
    featured = add_date_features(input_frame)
    return featured[REGRESSION_FEATURES]
