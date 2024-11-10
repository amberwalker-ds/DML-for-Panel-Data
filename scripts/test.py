import pandas as pd
from scripts.data_processing import *
from scripts.utils import *
from scripts.modeling import *

#load config
config = load_config("config.yaml")

# #access config values
# lag_type = config['lag_type']
# base_covariates = config['base_covariates']
# path_out = config['path_out']
# end_period = config['end_period']
# outcome_variable = config['outcome_variable']
# num_lags = config['num_lags']
# n_splits = config['n_splits']
# test_size = config['test_size']

#load and build a small test set with fake treatment data
def load_and_prepare_test_data():

    #load and preprocess data
    panel_df = load_clean_panel_data(config['panel_path'])
    pax_df = load_clean_pax_data(config['pax_path'], config['mapping_path'])

    #build pax features
    pax_df = build_pax_features(pax_df)

    #merge data
    df = merge_pax_panel_data(pax_df, panel_df)

    #impute missing data
    clean_df = impute_missing_data(df)
    
    #filter for countries with peace agreements
    countries_with_agreement = clean_df[clean_df['any_agreement'] == 1]['isocode'].unique()
    df_filtered = clean_df[clean_df['isocode'].isin(countries_with_agreement)]
    
    #subset data for time range 200501 - 201001
    df_filtered = df_filtered[(df_filtered['period'] >= 200501) & (df_filtered['period'] <= 201001)]
    
    #randomly set `any_agreement` to 1 for 30% of the filtered data (so there are a good amount of treatments)
    sample_indices = df_filtered.sample(frac=0.3, random_state=42).index
    df_filtered.loc[sample_indices, 'any_agreement'] = 1

    #feature engineering
    df_with_features = feature_engineering(df_filtered, 2)
    df_with_ohe, ohe_list = build_one_hot_dummies(df_with_features)
    
    return df_with_ohe, ohe_list


def run_test_model(df, config, ohe_list):
    # Set up models
    model_y = RandomForestRegressor(random_state=42)
    model_t = RandomForestClassifier(random_state=42)
    
    #DML model
    X_lagged, T, Y = prepare_and_run(
        df=df,
        end_period=config['end_period'],
        outcome_variable=config['outcome_variable'],
        num_lags=config['num_lags'],
        base_covariates=config['base_covariates'],
        ohe_list=ohe_list,
        n_splits=config['n_splits'],
        test_size=config['test_size'],
        model_y=model_y,
        model_t=model_t,
        path_out=config['path_out'],
        lag_type=config['lag_type'],
        X_list=config['X_list']
    )

    #results
    print("Test Results")
    print("Outcome Y:", Y.head())
    print("Treatment T:", T.head())
    print("Lagged Features X_lagged Sample:", X_lagged.head())

#main function to run the test
if __name__ == "__main__":

    df, ohe_list = load_and_prepare_test_data()

    run_test_model(df, config, ohe_list)
