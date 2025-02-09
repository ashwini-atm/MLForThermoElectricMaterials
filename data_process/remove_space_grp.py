from sklearn.base import BaseEstimator, TransformerMixin

class RemoveSpaceGroup(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.space_grp_feats = None
    
    def fit(self, X, y=None):
        if isinstance(X, tuple):
            X, _ = X
        self.space_grp_feats = [feat for feat in X.columns.values 
                                        if 'SpaceGroup' in feat]
        return self
    
    def transform(self, X, y=None):
        if isinstance(X, tuple):
            X, y = X
        return X.drop(self.space_grp_feats, axis = 1), y