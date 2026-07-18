import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X = pd.DataFrame(
    cancer.data,
    columns=cancer.feature_names
)


def get_sample():
    return X.iloc[[0]].copy()


def get_feature_names():
    return list(X.columns)