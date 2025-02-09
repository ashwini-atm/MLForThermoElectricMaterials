import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class HighCorrFeatRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        if 0.5 >= threshold <= 0.95:
            print("Threshold range [0.5-0.95]")
        self.threshold = max(min(threshold, 0.95), 0.5)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if isinstance(X, tuple):
            X, y = X
        corr = X.corr(method = 'spearman').abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr_feats = [column for column in upper.columns if any(upper[column] > 0.90)]

        return X.drop(high_corr_feats, axis=1), y