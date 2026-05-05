
# 📉 Customer Churn Prediction

A machine learning project that predicts customer churn using three classification algorithms: **Logistic Regression**, **Random Forest**, and **XGBoost**. The models are trained and evaluated on a customer dataset to identify customers who are likely to leave a service.

---

## 📁 Project Structure

```
churn-/
├── Logistic_regression/
│   └── train_test.py          # Logistic Regression model training & evaluation
├── Random_forest/
│   └── train_test.py          # Random Forest model training & evaluation
├── Xgboost/
│   └── train_test.py          # XGBoost model training & evaluation
├── data.py                    # Data loading, cleaning, and EDA
├── main.py                    # Entry point — runs all three models
├── customer_churn_data.csv    # Raw dataset
├── __init__.py
└── README.md
```

---

## 📊 Dataset

The dataset (`customer_churn_data.csv`) contains customer information including:

| Feature | Description |
|---|---|
| `Age` | Customer age (filtered to ≥ 18) |
| `Tenure` | Number of months the customer has been with the service |
| `MonthlyCharges` | Monthly billing amount |
| `TotalCharges` | Total amount charged (numeric, coerced if needed) |
| `ContractType` | Type of contract (e.g., Month-to-month, One year, Two year) |
| `Churn` | Target variable — `Yes`/`No` mapped to `1`/`0` |

---

## ⚙️ Data Pipeline (`data.py`)

The `load_and_clean_data()` function handles:

- Loading the CSV and reporting shape, missing values, and data types
- Encoding the `Churn` column (`Yes → 1`, `No → 0`)
- Forward-filling missing values
- Converting `TotalCharges` to numeric and imputing with the median
- Filtering out rows where `Age < 18` or `MonthlyCharges < 0`
- Printing key insights: churn rate by contract type, average tenure, and average monthly charges by churn status
- Plotting a pie chart of churn distribution

---

## 🤖 Models

Three models are implemented, each in their own module:

### 1. Logistic Regression (`Logistic_regression/train_test.py`)
A linear baseline classifier suitable for binary classification tasks.

### 2. Random Forest (`Random_forest/train_test.py`)
An ensemble of decision trees that reduces overfitting and handles non-linear relationships.

### 3. XGBoost (`Xgboost/train_test.py`)
A gradient boosting framework known for high performance on tabular data.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib scikit-learn xgboost
```

### Run

```bash
python main.py
```

This will sequentially train and evaluate all three models.

---

## 📈 Key Insights (from EDA)

- Churn rate varies significantly by **contract type** — month-to-month contracts tend to have higher churn.
- Customers who churned generally have **shorter tenure**.
- Churned customers tend to have **higher monthly charges** on average.

---

## 🛠️ Tech Stack

- **Python 3.x**
- **pandas** — data manipulation
- **matplotlib** — visualization
- **scikit-learn** — Logistic Regression, Random Forest, preprocessing & evaluation
- **XGBoost** — gradient boosting classifier

---

## 👤 Author

**Sanaa Zeroual** — [@sanaa-zeroual](https://github.com/sanaa-zeroual)
