
import pandas as pd

def load_data():
    path = "data/breast_cancer.csv"
    return pd.read_csv(path)
