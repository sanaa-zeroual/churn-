import sys
from pathlib import Path

from Logistic_regression.train_test import run_logistic_regression
from Random_forest.train_test import run_random_forest
from Xgboost.train_test import run_xgboost

if __name__ == "__main__":
    run_logistic_regression()
    run_random_forest()
    run_xgboost()
