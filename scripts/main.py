import pandas as pd
from data_processing import *
from utils import *
from modeling import *

#load config
config = load_config("config.yaml")

#access config values
lag_type = config['lag_type']
base_covariates = config['base_covariates']
path_out = config['path_out']
end_period = config['end_period']
outcome_variable = config['outcome_variable']
num_lags = config['num_lags']
n_splits = config['n_splits']
test_size = config['test_size']

#load and preprocess data
panel_df = load_clean_panel_data(config['panel_path'])
pax_df = load_clean_pax_data(config['pax_path'], config['mapping_path'])

#build pax features
pax_df = build_pax_features(pax_df)

#merge data
df = merge_pax_panel_data(pax_df, panel_df)

#impute missing data
clean_df = impute_missing_data(df)

# #feature engineering
df_with_features = feature_engineering(clean_df, 2)
df_with_ohe, ohe_list = build_one_hot_dummies(df_with_features)

# Create models from config
model_y = create_model(config['model_y'])
model_t = create_model(config['model_t'])

# Create X_list based on base_covariates
X_list = [f'{cov}_lag0' for cov in base_covariates]

# Run the prepare_and_run function
X_lagged, T, Y = prepare_and_run(
    df=df_with_ohe,
    end_period=end_period,
    outcome_variable=outcome_variable,
    num_lags=num_lags,
    base_covariates=base_covariates,
    ohe_list=ohe_list,
    n_splits=n_splits,
    test_size=test_size,
    model_y=model_y,
    model_t=model_t,
    path_out=path_out,
    lag_type=lag_type,
    X_list=X_list
)
