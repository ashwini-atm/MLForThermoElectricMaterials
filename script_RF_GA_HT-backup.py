import os
from utils import *
from datetime import datetime
# from sklearn.ensemble import RandomForestRegressor

date_time = datetime.now().strftime('%Y_%m_%d_%H_%M')
dataset_file = os.getenv("DATASET", "../df_300_magpie_drop_vif_std.csv")
thermoelectric_prop = os.getenv("PROPERTY", "seebeck_coefficient")
temperature = int(os.getenv("TEMPERATURE", 300))
estimator_choice = os.getenv("ESTIMATOR", "random_forest")
Estimator = estimators[estimator_choice]

mrl_file = os.getenv("MRL", "mrl.csv")
dataset_name = dataset_file.split('/')[-1].split('.')[0]
df, y = get_data(mrl_file, dataset_file, temperature, thermoelectric_prop)

print(11111111111111111)
#RF for default parameters
rf_score_base = get_score_from_estimator(Estimator, df, y)
rf_score_dict_base = {'temperature': temperature, 'property': thermoelectric_prop,
                    'dataset': dataset_name, 'estimator': Estimator.__name__, 'result_from': 'baseline'}
rf_score_dict_base.update(rf_score_base)

param_grid = param_grid_dict[estimator_choice]

print(22222222222222)
# Define a model with all features, best parameters before feature selection 
best_params_before_feat_selection = get_best_hyperparameter_for_param_grid(Estimator, param_grid, df, y)
rf_score_before_feat_selection = get_score_from_estimator(Estimator, df, y, **best_params_before_feat_selection)
rf_score_dict_before_feat_selection = {'temperature': temperature, 'property': thermoelectric_prop,
                                        'dataset': dataset_name, 'estimator': Estimator.__name__, 'result_from': 'before_feat_selection'}
rf_score_dict_before_feat_selection.update(rf_score_before_feat_selection)

print(3333333333333333)
# Running GA for feature selection
fitness_function = get_ga_fitness_function(Estimator, df, y)
feat_mask = get_feat_mask_from_ga(df.shape[1], fitness_function)

print(444444444444444)
# saving features selected from GA
selected_features = np.array(list(df.columns))[feat_mask]
feat_out_file = f"{dataset_file}_{estimator_choice}_{temperature}_{thermoelectric_prop}_features_from_ga_{date_time}.txt"
np.savetxt(feat_out_file, selected_features, fmt='%s')

print(555555555555555)
# Define a model with features from GA, best parameters after feature selection
df_ga_feat = df[selected_features.tolist()]
best_params_after_feat_selection = get_best_hyperparameter_for_param_grid(Estimator, param_grid, df_ga_feat, y)
rf_score_after_feat_selection = get_score_from_estimator(Estimator, df_ga_feat, y, **best_params_after_feat_selection)
rf_score_dict_after_feat_selection = {'temperature': temperature, 'property': thermoelectric_prop,
                                        'dataset': dataset_name, 'estimator': Estimator.__name__, 'result_from': 'after_feat_selection'}
rf_score_dict_after_feat_selection.update(rf_score_after_feat_selection)

print(666666666666666)
# saving results to CSV
results_df = pd.DataFrame([rf_score_dict_base, rf_score_dict_before_feat_selection, rf_score_dict_after_feat_selection])
out_file_name = f"{dataset_file}_{estimator_choice}_{temperature}_{thermoelectric_prop}_results_{date_time}.csv"
results_df.to_csv(out_file_name, index=False)