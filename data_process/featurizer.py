import pandas as pd
from matminer.featurizers.conversions \
                import StrToComposition
from matminer.featurizers.composition import ElementProperty
from sklearn.base import BaseEstimator, TransformerMixin

class Featurizer(BaseEstimator, TransformerMixin):
    def __init__(self, temp=None, y_name="seebeck_coefficient"):
        self.temp = temp
        self.y_name = y_name
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if self.temp:
            X = X[X['temperature']==self.temp]
        y = X[self.y_name]
        stc = StrToComposition()
        ef = ElementProperty.from_preset(preset_name='magpie')
        X = pd.DataFrame(X["Formula"])
        X = stc.featurize_dataframe(X, "Formula", pbar=False)
        X = ef.featurize_dataframe(X, col_id='composition', pbar=False)
        return X.drop(["Formula", "composition"], axis=1), y