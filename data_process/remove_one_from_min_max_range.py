from sklearn.base import BaseEstimator, TransformerMixin

class MinMaxOrRangeRemover(BaseEstimator, TransformerMixin):
    def __init__(self, feat_to_remove="maximum"):
        self.feat_to_remove = feat_to_remove
        if feat_to_remove not in ["maximum", "minimum", "range"]:
            print(f"{feat_to_remove} is invalid. Removing 'maximum'.")
            self.feat_to_remove = "maximum"
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if isinstance(X, tuple):
            X, y = X
        max_col_names = [feat for feat in X.columns if 'maximum' in feat]
        return X.drop(max_col_names, axis=1), y