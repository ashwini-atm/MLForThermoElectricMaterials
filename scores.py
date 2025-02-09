from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error as mae 
from sklearn.metrics import mean_squared_error as mse 
from sklearn.metrics import r2_score as r2
from scipy.stats import pearsonr as r

import numpy as np

# Function for relative absolute error (RAE), and relative squared error (RSE):

def rae(y_true,y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    num = np.abs(y_pred - y_true).sum()
    denom = np.abs(y_true - y_true.mean()).sum()
    return num/denom 

def rse(y_true,y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    num = ((y_pred - y_true)**2).sum()
    denom = ((y_true - y_true.mean())**2).sum()
    return num/denom

def get_score(y_true, y_pred):
    scores = {}
    scores['R'] = r(y_true, y_pred)[0] # by default it returns PCC(correlation) and p-value(hypothesis testing)
    scores['R2'] = r2(y_true, y_pred)
    scores['MAE'] = mae(y_true, y_pred)
    scores['RMSE'] = mse(y_true, y_pred)**0.5
    scores['RAE'] = rae(y_true, y_pred)
    scores['RSE'] = rse(y_true, y_pred)
    return scores