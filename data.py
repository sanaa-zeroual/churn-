import pandas as pd
import matplotlib.pyplot as plt

def load_and_clean_data():
    df = pd.read_csv("customer_churn_data.csv")

    print("Shape:", df.shape)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nData types:")
    print(df.dtypes)

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    if df['Churn'].mean() > 0.5:  
        df['Churn'] = 1 - df['Churn']

    df.ffill(inplace=True)

    if 'TotalCharges' in df.columns and df['TotalCharges'].dtype == 'object':
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    df = df[df['Age'] >= 18]
    df = df[df['MonthlyCharges'] >= 0]

    print("Key Insights:\n")
    print("Churn by Contract: \n")
    print(df.groupby("ContractType")["Churn"].mean() * 100)

    print("Tenure difference:\n")
    print(df.groupby("Churn")["Tenure"].mean())

    print("Charges difference: \n")
    print(df.groupby("Churn")["MonthlyCharges"].mean())

    df['Churn'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Churn Distribution')
    plt.show()

    print(f"\n Cleaned data: {df.shape} rows, {df.shape[1]} columns")
    print(f"Churn rate: {df['Churn'].mean()*100:.1f}%")

    return df
