import re
from sklearn.base import BaseEstimator, TransformerMixin

class LowVarFeatRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        if 0.5 >= threshold <= 0.95:
            print("Threshold range [0.5-0.95]")
        self.threshold = max(min(threshold, 0.95), 0.5)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if isinstance(X, tuple):
            X, y = X
        float_feats = [feat for feat in X.columns[1:]
                                        if feat.split(' ')[1] in ['mean', 'avg_dev']]
        properties = set([prop.split(" ")[-1] for prop in X.columns.values])
        temp_feat = set(X) - set(float_feats)

        int_feat = [feat for feat in temp_feat \
            if feat.split(' ')[-1] in \
                    [prop for prop in properties if re.match(r'(temperature|Number|Row|Column|.*Valence|.*Unfilled)', prop)]]

        float_feats += list(temp_feat - set(int_feat))

        assert len(float_feats) + len(int_feat) == len(X.columns)
        X[int_feat] = X[int_feat].astype('int')
        X_int = X.select_dtypes('int')
        X_float = X.select_dtypes('float')
        feats_to_drop = [feat for feat in X_int.columns if \
                                            X_int[feat].value_counts(normalize=True).max() > self.threshold]

        feats_to_drop += [feat for feat in X_float.columns if \
                                        X_float[feat].value_counts(normalize=True).max() > self.threshold]

        return X.drop(feats_to_drop, axis=1), y