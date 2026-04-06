from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from train import run_training_pipeline
from utils import (
    ANOMALY_MODEL_PATH,
    CATEGORIES,
    CLASSIFICATION_MODEL_PATH,
    DATA_PATH,
    METRICS_PATH,
    PAYMENT_METHODS,
    PLOTS_DIR,
    REGRESSION_MODEL_PATH,
    clean_expense_data,
    create_regression_input,
    get_budget_suggestion,
    get_top_spending_categories,
    load_or_create_dataset,
    load_pickle,
    prepare_anomaly_features,
)


st.set_page_config(page_title="Smart Personal Expense Tracker", page_icon=":money_with_wings:", layout="wide")


@st.cache_data(show_spinner=False)
def get_expense_data() -> pd.DataFrame:
    return load_or_create_dataset()


@st.cache_resource(show_spinner=False)
def get_artifacts() -> dict:
    return {
        "regressor": load_pickle(REGRESSION_MODEL_PATH),
        "classifier": load_pickle(CLASSIFICATION_MODEL_PATH),
        "anomaly_artifact": load_pickle(ANOMALY_MODEL_PATH),
        "metrics": load_pickle(METRICS_PATH),
    }


def artifacts_exist() -> bool:
    return all(
        path.exists()
        for path in [REGRESSION_MODEL_PATH, CLASSIFICATION_MODEL_PATH, ANOMALY_MODEL_PATH, METRICS_PATH]
    )


def show_top_metrics(dataframe: pd.DataFrame) -> None:
    total_spend = dataframe["amount"].sum()
    avg_spend = dataframe["amount"].mean()
    budget_suggestion = get_budget_suggestion(dataframe)
    top_categories = get_top_spending_categories(dataframe, top_n=3)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Spend", f"Rs. {total_spend:,.2f}")
    col2.metric("Average Expense", f"Rs. {avg_spend:,.2f}")
    col3.metric("Suggested Budget", f"Rs. {budget_suggestion:,.2f}")

    st.subheader("Top 3 Highest Spending Categories")
    for category, amount in top_categories.items():
        st.write(f"- {category}: Rs. {amount:,.2f}")


def plot_monthly_trends(dataframe: pd.DataFrame) -> None:
    working = dataframe.copy()
    working["date"] = pd.to_datetime(working["date"])
    working["year_month"] = working["date"].dt.to_period("M").astype(str)
    monthly_spend = working.groupby("year_month")["amount"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.lineplot(data=monthly_spend, x="year_month", y="amount", marker="o", ax=ax)
    ax.set_title("Monthly Expense Trends")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Spend")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)


def plot_category_spending(dataframe: pd.DataFrame) -> None:
    category_spend = dataframe.groupby("category")["amount"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.barplot(
        x=category_spend.index,
        y=category_spend.values,
        hue=category_spend.index,
        dodge=False,
        legend=False,
        palette="crest",
        ax=ax,
    )
    ax.set_title("Category-wise Spending")
    ax.set_xlabel("Category")
    ax.set_ylabel("Total Spend")
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)


def plot_prediction_vs_actual(metrics: dict) -> None:
    actual = metrics["regression"]["actual"][:60]
    predicted = metrics["regression"]["predicted"][:60]
    sample_index = list(range(1, len(actual) + 1))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(sample_index, actual, label="Actual", linewidth=2)
    ax.plot(sample_index, predicted, label="Predicted", linewidth=2)
    ax.set_title("Prediction vs Actual")
    ax.set_xlabel("Test Sample")
    ax.set_ylabel("Amount")
    ax.legend()
    st.pyplot(fig)


st.title("Smart Personal Expense Tracker using Machine Learning")
st.caption(
    "Track expenses, predict future spending, auto-categorize text descriptions, and detect unusual transactions."
)

if not artifacts_exist():
    st.warning("No trained models were found yet. Generate the sample dataset and train the ML models first.")
    if st.button("Generate Dataset and Train Models", type="primary"):
        with st.spinner("Training models and preparing visualizations..."):
            run_training_pipeline(force_generate=not DATA_PATH.exists())
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Training completed successfully. The dashboard is ready to use.")
        st.rerun()
    st.stop()

artifacts = get_artifacts()
regressor = artifacts["regressor"]
classifier = artifacts["classifier"]
anomaly_artifact = artifacts["anomaly_artifact"]
metrics = artifacts["metrics"]

expense_df = clean_expense_data(get_expense_data())
anomaly_features, _ = prepare_anomaly_features(expense_df, anomaly_artifact["feature_columns"])
anomaly_predictions = anomaly_artifact["model"].predict(anomaly_features)
anomaly_scores = anomaly_artifact["model"].decision_function(anomaly_features)

expense_df["anomaly_flag"] = (anomaly_predictions == -1).astype(int)
expense_df["anomaly_score"] = anomaly_scores

if "expense_feedback" in st.session_state:
    feedback = st.session_state.pop("expense_feedback")
    st.success(feedback)

show_top_metrics(expense_df)

overview_tab, add_tab, predict_tab, anomaly_tab = st.tabs(
    ["Overview", "Add Expense", "Future Prediction", "Anomaly Detection"]
)

with overview_tab:
    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", metrics["regression"]["mae"])
    col2.metric("RMSE", metrics["regression"]["rmse"])
    col3.metric("Classification Accuracy", f"{metrics['classification']['accuracy'] * 100:.2f}%")

    plot_monthly_trends(expense_df)
    plot_category_spending(expense_df)
    plot_prediction_vs_actual(metrics)

    st.subheader("Saved Charts")
    st.write(
        f"- Monthly trends: `{Path(PLOTS_DIR / 'monthly_expense_trends.png')}`\n"
        f"- Category spending: `{Path(PLOTS_DIR / 'category_spending.png')}`\n"
        f"- Prediction vs actual: `{Path(PLOTS_DIR / 'prediction_vs_actual.png')}`\n"
        f"- Confusion matrix: `{Path(PLOTS_DIR / 'confusion_matrix.png')}`"
    )

with add_tab:
    st.subheader("Add a New Expense")
    st.write("The category is predicted automatically from the description using the NLP classifier.")

    with st.form("add_expense_form"):
        expense_date = st.date_input("Expense Date")
        amount = st.number_input("Amount", min_value=0.0, step=10.0)
        description = st.text_input("Description", placeholder="Example: Swiggy order")
        payment_method = st.selectbox("Payment Method", PAYMENT_METHODS)
        submit_expense = st.form_submit_button("Add Expense")

    if submit_expense:
        description = description.strip() or "miscellaneous expense"
        predicted_category = classifier.predict([description])[0]

        new_expense = pd.DataFrame(
            [
                {
                    "date": str(expense_date),
                    "amount": float(amount),
                    "category": predicted_category,
                    "description": description,
                    "payment_method": payment_method,
                }
            ]
        )

        new_anomaly_features, _ = prepare_anomaly_features(new_expense, anomaly_artifact["feature_columns"])
        is_anomaly = anomaly_artifact["model"].predict(new_anomaly_features)[0] == -1

        updated_data = pd.concat([get_expense_data(), new_expense], ignore_index=True)
        updated_data.to_csv(DATA_PATH, index=False)

        anomaly_text = "This expense looks unusual." if is_anomaly else "This expense looks normal."
        st.session_state["expense_feedback"] = (
            f"Expense added successfully. Predicted category: {predicted_category}. {anomaly_text}"
        )
        st.cache_data.clear()
        st.rerun()

    st.subheader("Quick Category Prediction")
    quick_text = st.text_input("Try a description", placeholder="Example: movie tickets")
    if quick_text:
        quick_prediction = classifier.predict([quick_text])[0]
        st.info(f"Predicted category: {quick_prediction}")

with predict_tab:
    st.subheader("Predict Future Expense")
    st.write("Choose a future date, category, and payment method to estimate a likely expense amount.")

    future_date = st.date_input("Future Expense Date")
    future_category = st.selectbox("Expected Category", CATEGORIES)
    future_payment_method = st.selectbox("Expected Payment Method", PAYMENT_METHODS, key="future_payment")

    if st.button("Predict Expense Amount", type="primary"):
        input_frame = create_regression_input(
            expense_date=str(future_date),
            category=future_category,
            payment_method=future_payment_method,
        )
        predicted_amount = regressor.predict(input_frame)[0]
        st.success(f"Predicted future expense: Rs. {predicted_amount:,.2f}")

with anomaly_tab:
    st.subheader("Detected Unusual Spending")
    anomaly_rows = expense_df[expense_df["anomaly_flag"] == 1].sort_values("amount", ascending=False)

    st.write(f"Total anomalies detected in current dataset: **{len(anomaly_rows)}**")
    st.dataframe(
        anomaly_rows[["date", "amount", "category", "description", "payment_method"]].head(20),
        use_container_width=True,
    )

    st.subheader("Anomaly Scatter View")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(pd.to_datetime(expense_df["date"]), expense_df["amount"], alpha=0.5, label="Normal")

    if not anomaly_rows.empty:
        ax.scatter(
            pd.to_datetime(anomaly_rows["date"]),
            anomaly_rows["amount"],
            color="red",
            label="Anomaly",
        )

    ax.set_title("Anomaly Detection Across Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Amount")
    ax.legend()
    st.pyplot(fig)
