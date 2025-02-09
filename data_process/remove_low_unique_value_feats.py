from sklearn.base import BaseEstimator, TransformerMixin

class LowUniqueValueFeatRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=2):
        self.threshold = threshold
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if isinstance(X, tuple):
            X, y = X
        return X.drop(X.columns[X.nunique()<self.threshold], axis=1), y