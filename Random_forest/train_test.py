from data import load_and_clean_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def run_random_forest():
    df = load_and_clean_data()

    y = df["Churn"]
    X = df.drop("Churn", axis=1)
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print(f"\nRandom Forest Accuracy: {accuracy*100:.2f}%")
