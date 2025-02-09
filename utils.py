import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import GridSearchCV, cross_val_predict

from geneal.genetic_algorithms import BinaryGenAlgSolver

from scores import get_score

estimators = {
    "random_forest": RandomForestRegressor,
    "svr": SVR
}

param_grid_dict = {
    "random_forest": {'bootstrap': [True, False],
                        'max_depth': [2, 3, 5, 10, 20, 30, 50, None],
                        'max_features': ['auto', 'sqrt'],
                        'min_samples_leaf': [1, 2, 4],
                        'min_samples_split': [2, 5, 10],
                        'n_estimators': [100, 200, 500, 600, 800, 1000, 1200, 1400, 1500]
                    }
}

# mae scorer for hyperparam tuning
mae_scorer = make_scorer(mae, greater_is_better=False)

def get_data(mrl_file, dataset_file, temperature, thermoelectric_prop):
    df_mrl = pd.read_csv(mrl_file)
    df_mrl = df_mrl[df_mrl['temperature']==temperature]
    y = df_mrl[thermoelectric_prop]
    df = pd.read_csv(dataset_file)
    return df, y

def get_score_from_estimator(Estimator, df, y, **kwargs):
    #RF for default parameters
    estimator = Estimator(**kwargs)
    y_pred_base = cross_val_predict(estimator, df, y, cv=5)
    rf_score_base = get_score(y, y_pred_base)
    return rf_score_base

def get_best_hyperparameter_for_param_grid(Estimator, param_grid, df, y):
    estimator = Estimator()
    grid_search_before_feat_selection = GridSearchCV(estimator = estimator, param_grid = param_grid,
                                                cv = 5, verbose=0, n_jobs = -1, scoring=mae_scorer) # Fit the random search model

    _ = grid_search_before_feat_selection.fit(df, y)
    best_params_before_feat_selection = grid_search_before_feat_selection.best_params_
    return best_params_before_feat_selection

    #rf_score_before_feat_selection = get_score_from_rf_model(df, y)

def get_ga_fitness_function(Estimator, df, y_inp):
    def ga_fitness_function(chromosome):
        feat_mask = chromosome.astype(bool)
        if any(feat_mask):
            X = pd.DataFrame(df.iloc[:, feat_mask])
            y = y_inp
            estimator = Estimator()
            y_pred = cross_val_predict(estimator, X, y, cv=3)
            return -mae(y, y_pred)
        else:
            return -np.inf
    return ga_fitness_function

def get_feat_mask_from_ga(n_genes, fitness_function):
    solver = BinaryGenAlgSolver(
        n_genes=n_genes, #df.shape[1], 
        fitness_function=fitness_function,
        pop_size=50,
        max_gen=100,
        mutation_rate=0.2,
        selection_rate=0.6,
        selection_strategy="roulette_wheel",
        verbose=False,
        show_stats=False,
        plot_results=False
    )
    solver.solve()
    feat_mask = solver.best_individual_.astype(bool)
    return feat_mask