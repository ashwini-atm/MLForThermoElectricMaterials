import pandas as pd
from sklearn.preprocessing import StandardScaler as StdScaler
from sklearn.base import BaseEstimator, TransformerMixin

class StandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if isinstance(X, tuple):
            X, y = X
        scaler = StdScaler()
        X_trans = scaler.fit_transform(X)
        return pd.DataFrame(X_trans, columns=X.columns), y