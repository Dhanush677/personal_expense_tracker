# Smart Personal Expense Tracker using Machine Learning

This project is a beginner-friendly end-to-end machine learning application that:

- predicts future expenses
- categorizes expense descriptions automatically
- detects unusual spending patterns
- suggests a budget based on recent spending
- shows the top 3 spending categories

## Project Structure

```text
Expense_tracker/
├── app.py
├── train.py
├── utils.py
├── requirements.txt
├── data/
│   ├── expenses.csv
│   └── plots/
└── models/
    ├── expense_regressor.pkl
    ├── category_classifier.pkl
    ├── anomaly_detector.pkl
    └── metrics.pkl
```

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit

## Machine Learning Models

1. Expense Prediction
- Model: `RandomForestRegressor`
- Goal: Predict future expense amount

2. Automatic Categorization
- Model: `TF-IDF + LogisticRegression`
- Goal: Predict category from description text

3. Anomaly Detection
- Model: `IsolationForest`
- Goal: Flag unusual spending behaviour

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train models and generate sample data:

```bash
python train.py
```

3. Start the Streamlit app:

```bash
python -m streamlit run app.py
```

## Notes

- If no dataset is available, `train.py` creates a synthetic dataset with 1500 rows automatically.
- Trained models are saved inside the `models/` folder as `.pkl` files.
- Visualizations are saved inside `data/plots/`.
- You can also train models directly from the Streamlit app by clicking the training button.
