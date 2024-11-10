import pandas as pd
import numpy as np
import pickle

def load_clean_panel_data(filepath):
    """
    Load the panel data from path, drop first col
    and only fetch data with population more than 0
    
    Parameters:
        filepath (str): Path to the panel data CSV file.

    Returns:
        pd.DataFrame: A clean DataFrame containing the panel data.
    
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df['period'] = (df.year.astype(str) + df.month.astype(str).apply(lambda x: '0' + x if len(x) == 1 else x)).astype(int)

    return df

def load_clean_pax_data(pax_path, mapping_path):
    """
    Cleans and processes the pax dataset.

    Parameters:
        pax_path (str): Path to the PA_X CSV file.
        mapping_file (str): Path to the pickled dictionary mapping peace process
        names tp ISO codes.

    Returns:
        pd.DataFrame: A cleaned DataFrame with ISO codes and 'period' column.
    """
    df = pd.read_csv(pax_path)
    df.columns = df.columns.str.lower()

    with open(mapping_path, 'rb') as file:
        peace_process_to_iso = pickle.load(file)

    df['isocode'] = df['peace process name'].map(peace_process_to_iso)

    # Dictionary mapping country strings to lists of ISO country codes
    country_to_iso_codes = {
        'Angola/Burundi/Central African Republic/Democratic Republic of Congo/Kenya/Republic of Congo/Rwanda/Sudan/Tanzania/Uganda/Zambia/(African Great Lakes)': 
            ['AGO', 'BDI', 'CAF', 'COD', 'KEN', 'COG', 'RWA', 'SDN', 'TZA', 'UGA', 'ZMB'],
        'Angola/Burundi/Central African Republic/Democratic Republic of Congo/Republic of Congo/Rwanda/South Africa/South Sudan/Tanzania/Uganda/Zambia/(African Great Lakes)': 
            ['AGO', 'BDI', 'CAF', 'COD', 'COG', 'RWA', 'ZAF', 'SSD', 'TZA', 'UGA', 'ZMB'],   
        'Angola/Cabinda': 
            ['AO-CAB'],  # Custom code for Cabinda, subdivision of Angola   
        'Angola/Central African Republic/Democratic Republic of Congo/Kenya/Republic of Congo/Rwanda/Sudan/Tanzania/Uganda/Zambia/(African Great Lakes)': 
            ['AGO', 'CAF', 'COD', 'KEN', 'COG', 'RWA', 'SDN', 'TZA', 'UGA', 'ZMB'],  
        'Croatia/Slovenia/Yugoslavia (former)': 
            ['HRV', 'SVN', 'YUG'],   
        'Croatia/Yugoslavia (former)': 
            ['HRV', 'YUG'], 
        'Kurds-Kurdistan': 
            ['KRG'],  # Custom code for Kurdistan region 
        'Slovenia/Yugoslavia (former)': 
            ['SVN', 'YUG']
    }

    df.loc[df['isocode'].isna(), 'isocode'] = df['country'].map(country_to_iso_codes)

    clean_df = df.dropna(subset='isocode')

    clean_df.loc[:, 'signed_date'] = pd.to_datetime(clean_df['signed date'])
    clean_df.loc[:, 'period'] = clean_df['signed_date'].dt.strftime('%Y%m').astype(int)

    return clean_df

def build_pax_features(df):
    """
    Builds new columns in the pax dataset.

    Parameters:
        df: PA_X DataFrame

    Returns:
        pd.DataFrame with new columns:
            any_agreement: indicated whether a PA is present for period and country
            num_agreements counts the number of peace agreements that are in place for period and country
            agreement_id: lists the ID for each peace agreement that is in place for that period and country
    """
    filtered_df = df[['agreementid', 'isocode', 'period', 'name']]
    filtered_df['isocode'] = filtered_df['isocode'].apply(lambda x: x if isinstance(x, list) else [x])
    df_exploded = filtered_df.explode('isocode')

    df_exploded['pa_present'] = 1

    df_aggregated = df_exploded.groupby(['isocode', 'period']).agg(
        any_agreement=('pa_present', 'max'),
        num_agreements = ('pa_present', 'sum'),
        agreement_id = ('agreementid', lambda x: ', '.join(map(str, x.unique())))
    ).reset_index()
    
    return df_aggregated

def merge_pax_panel_data(pax_df, panel_df):
    """
    Merges the Pax data onto the panel data by doing a left join, by the period and isocode

    Parameters:
        pax_df: DataFrame for the pa-x data
        panel_df: DataFrame for the panel data

    Returns:
        pd.DataFrame: DataFrame containing both the pax and panel data
    """
    df = panel_df.merge(pax_df, on=['isocode', 'period'], how='left')
    return df


def impute_missing_data(df):
    """
    Impute missing data by:
    1. Grouping data by 'isocode' and filling specified columns with the group's mean.
    2. Forward-filling specified columns within each 'isocode' and 'year' group.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing pax and panel data.

    Returns:
        pd.DataFrame: The DataFrame with missing data imputed.
    """
    columns_to_impute = [
        'tokens', 'obs', 'sentiment_words', 'uncertainty', 'ste_theta0', 'ste_theta1', 'ste_theta2', 'ste_theta3',
        'ste_theta4', 'ste_theta5', 'ste_theta6', 'ste_theta7', 'ste_theta8', 'ste_theta9', 'ste_theta10', 'ste_theta11',
        'ste_theta12', 'ste_theta13', 'ste_theta14', 'ste_theta0_stock', 'ste_theta1_stock', 'ste_theta2_stock',
        'ste_theta3_stock', 'ste_theta4_stock', 'ste_theta5_stock', 'ste_theta6_stock', 'ste_theta7_stock', 'ste_theta8_stock',
        'ste_theta9_stock', 'ste_theta10_stock', 'ste_theta11_stock', 'ste_theta12_stock', 'ste_theta13_stock', 'ste_theta14_stock',
        'sentiment_stock', 'uncertainty_index', 'democracy0', 'democracy1', 'democracy2', 'democracy3', 'democracy4', 'democracy5'
    ]

    for col in columns_to_impute:
        # Convert columns with strings of numbers separated by spaces to a single numeric value
        df[col] = df[col].apply(lambda x: np.mean([float(num) for num in str(x).split()]) if isinstance(x, str) else x)
        # Convert to numeric, coercing any non-numeric values to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    columns_to_ffill = ['childmortality', 'populationwb']
    
    # Impute missing data in specified columns with the mean for each 'isocode' group
    imputed_means = df.groupby(['isocode'])[columns_to_impute].transform(lambda x: x.fillna(x.mean()))
    df[columns_to_impute] = imputed_means

    # Forward fill remaining NaNs within each 'isocode' and 'year' group
    for col in columns_to_ffill:
        df[col] = df.groupby(['isocode', 'year'])[col].ffill()
    
    return df


def count_since_target_one(x: pd.Series):
    """
    Counts the number of periods since the last time the target variable was 1.
    If starting in a period of peace (0), it begins counting from 1.
    """
    # Convert to a list and initialize result list
    x = list(x)
    if len(x) == 0:  # Handle empty list case
        return []

    y = []
    for n in range(len(x)):
        if x[n] == 0:
            if n == 0:  # Start with peace
                y.append(1)
            else:
                y.append(y[n-1] + 1)
        elif x[n] == 1:
            y.append(0)  # Reset on conflict
        else:
            y.append(np.nan)  # Handle any unexpected values
    return y


def count_until_target(series: pd.Series):
    series = series[::-1]

    # Create groups based on consecutive ones
    series = series == 0
    groups = (series == 0).cumsum()

    # Calculate cumulative sum within each group
    result = series.groupby(groups).cumsum()[::-1]

    result[groups == 0] = np.nan

    return result

def onset_in_target_zero(x:pd.Series, p:int):
    '''previously called onset_in_peace'''
    '''x is the original observation of target (e.g. coup) and y is an encoding of x to be a forecasting target (ons_coup_12)'''
    '''y signals if there will be an x=1 in the next p periods when the preceding period is x=0'''
    '''y = np.nan in periods of x=1 (e.g. when forecasting conflict onset, all periods in conflict are coded as np.nan)'''
    '''y = 0 if there will be x_{t+i}=0 for all i=1...p (for p periods forward) during x_{t}=0 (e.g. no conflict onset after a peaceful period)'''
    '''y = 1 if there will x_{t+i}=1 for at least one i=1...p (for p periods forward) during x_{t}=0'''
    x = list(x)
    y = []
    for i in range(len(x)):
        i0 = i+1
        i1 = i0+p
        if i1 <= len(x) and x[i]==0:
            y.append(np.max(x[i0:i1])*1)
        else:
            y.append(np.nan)
    return y

def feature_engineering(df, lags):
    """
    Adds required features:
        - Builds 'bestpc' variable which is the best estimate of deaths normalized by population
        - Builds 'since' violence and 'until' violence variables
        - Create lagged treatment and lagged X variables
        - Builds target variable by taking the log of violence

    Parameters:
        df (pd.DataFrame): the clean df with pax and panel data
        lags (int): the numbers of lags you would like to shift the data by

    Returns:
        pd.DataFrame: new df with added features
    """
    term_thresholds = ['anyviolence', 'armedconf', 'civilwar']
    covariates = ['since_anyviolence', 'since_armedconf', 'since_civilwar', 'tokens', 'obs', 'sentiment_words', 'uncertainty',
                  'ste_theta0', 'ste_theta1', 'ste_theta2', 'ste_theta3', 'ste_theta4', 'ste_theta5', 'ste_theta6', 
                  'ste_theta7', 'ste_theta8', 'ste_theta9', 'ste_theta10', 'ste_theta11', 'ste_theta12', 'ste_theta13', 
                  'ste_theta14', 'ste_theta0_stock', 'ste_theta1_stock', 'ste_theta2_stock', 'ste_theta3_stock', 
                  'ste_theta4_stock', 'ste_theta5_stock', 'ste_theta6_stock', 'ste_theta7_stock', 'ste_theta8_stock', 
                  'ste_theta9_stock', 'ste_theta10_stock', 'ste_theta11_stock', 'ste_theta12_stock', 'ste_theta13_stock', 
                  'ste_theta14_stock', 'sentiment_stock', 'populationwb']
    
    # Set bestpc variable
    df['best'] = df['best'].fillna(0)
    df['bestpc'] = df['best'] * 100000 / df['populationwb']

    # Build since and until variables
    for term in term_thresholds:
        df[f'since_{term}'] = df.groupby(['isocode'])[term].transform(count_since_target_one)
        if term == 'armedconf':
            df[f'until_{term}'] = df.groupby(['isocode'])[term].transform(count_until_target)

    df['any_agreement'] = df['any_agreement'].fillna(0)
    df['since_agreement'] = df.groupby(['isocode'])['any_agreement'].transform(count_since_target_one)
    df['until_agreement'] = df.groupby(['isocode'])['any_agreement'].transform(count_until_target)

    # Sort by country and period
    df.sort_values(by=['isocode', 'period'], inplace=True)

    # Create lagged variables for 'any_agreement' within each country
    for i in range(1, lags + 1):
        df[f'pa_lag{i}'] = df.groupby('isocode')['any_agreement'].shift(i)
    # Shift treatment variable backward for placebo test
    for i in range(1, lags + 1):
        df[f'pa_placebo_lag{i}'] = df.groupby('isocode')['any_agreement'].shift(-i)

    # Fill missing lag values with 0
    df[[f'pa_lag{i}' for i in range(1, lags + 1)]] = df[[f'pa_lag{i}' for i in range(1, lags + 1)]].fillna(0)
    df[[f'pa_placebo_lag{i}' for i in range(1, lags + 1)]] = df[[f'pa_placebo_lag{i}' for i in range(1, lags + 1)]].fillna(0)

    df = df.rename(columns={'any_agreement': 'pa_lag0'})

    # Drop rows with missing population
    df = df.dropna(subset=['populationwb'])
    df.sort_values(by=['isocode', 'period'], inplace=True)

    # Create lags for covariates
    number_of_lags = lags + 1
    new_columns = {}

    for cov in covariates:
        for lag in range(1, number_of_lags):
            new_columns[f'{cov}_lag{lag}'] = df.groupby('isocode')[cov].shift(lag)
            new_columns[f'{cov}_placebo_lag{lag}'] = df.groupby('isocode')[cov].shift(-lag)

    # Concatenate new lagged columns and fill any NaNs with 0
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1).fillna(0)

    # Rename original covariates with _lag0 suffix
    rename_dict = {col: f"{col}_lag0" for col in covariates}
    df = df.rename(columns=rename_dict)

    # Build target variable
    violence_intensity = np.log(df['bestpc'] + 1).fillna(0)
    df = pd.concat([df, violence_intensity.rename('violence_intensity')], axis=1)

    return df


def build_one_hot_dummies(df):
    """
    Builds a cleaned DataFrame and applies one-hot encoding to the 'isocode' column for country-specific fixed effects.

    Parameters:
    df : pandas.DataFrame
        The input dataframe containing all variables, including 'isocode'.

    Returns:
    tuple : (pd.DataFrame, list)
        - The cleaned and one-hot encoded DataFrame.
        - A list of the names of the one-hot encoded columns.
    """
    #columns to drop for X variable selection
    columns_to_drop = ['isocode', 'period', 'year', 'month', 'violence_intensity', 'ons_anyviolence_6', 
                       'ons_armedconf_6', 'ons_civilwar_6', 'num_agreements', 'agreement_id', 'since_agreement',
                       'until_agreement', 'uncertainty_index',  'democracy0', 'until_armedconf', 'democracy1', 
                       'democracy2', 'democracy3', 'democracy4', 'democracy5', 'best', 'ged_best_ns', 'ged_best_os', 
                       'ged_best_sb', 'anyviolence', 'armedconf', 'civilwar', 'past6', 'past12', 'past60', 
                       'past120', 'anyviolence_dp', 'armedconf_dp','civilwar_dp', 'ged_best_ns_dp', 
                       'ged_best_os_dp', 'ged_best_sb_dp', 'ons_ged_best_ns1', 'ons_ged_best_ns3', 
                       'ons_ged_best_ns12', 'ons_ged_best_os1', 'ons_ged_best_os12', 'ons_ged_best_sb1', 
                       'ons_ged_best_sb3', 'ons_ged_best_sb12', 'anyviolence1', 'armedconf1', 'civilwar1', 
                       'contig_anyviolence', 'pa_lag0', 'pa_lag1', 'pa_lag2', 'pa_lag3', 'pa_lag4', 
                       'pa_lag5', 'pa_lag6', 'pa_lag7', 'pa_lag8', 'pa_lag9', 'pa_lag10', 'pa_lag11', 'pa_lag12']
    
    #drop unnecessary columns to isolate X variables
    X_var = df.drop(columns=columns_to_drop, errors='ignore')
    cov_list = X_var.columns.to_list()

    #setup train data by dropping additional columns not needed for DML
    additional_drops = ['year', 'month', 'ons_anyviolence_6', 'ons_armedconf_6', 'ons_civilwar_6', 
                        'num_agreements', 'agreement_id', 'since_agreement', 'until_agreement', 
                        'uncertainty_index', 'democracy0', 'until_armedconf', 'democracy1', 'democracy2', 
                        'democracy3', 'democracy4', 'democracy5', 'best', 'ged_best_ns', 'ged_best_os', 
                        'ged_best_sb', 'anyviolence', 'armedconf', 'civilwar', 'past6', 'past12', 
                        'past60', 'past120', 'anyviolence_dp', 'armedconf_dp', 'civilwar_dp', 
                        'ged_best_ns_dp', 'ged_best_os_dp', 'ged_best_sb_dp', 'ons_ged_best_ns1', 
                        'ons_ged_best_ns3', 'ons_ged_best_ns12', 'ons_ged_best_os1', 'ons_ged_best_os12', 
                        'ons_ged_best_sb1', 'ons_ged_best_sb3', 'ons_ged_best_sb12', 'anyviolence1', 
                        'armedconf1', 'civilwar1', 'contig_anyviolence']
    train_data = df.drop(columns=additional_drops, errors='ignore')
    
    #fill missing values in covariates with 0
    train_data[cov_list] = train_data[cov_list].fillna(0)
    train_data_clean = train_data.dropna().reset_index(drop=True)

    #one-hot encoding to the 'isocode' column
    one_hot_encoded = pd.get_dummies(train_data_clean['isocode'], prefix='isocode', drop_first=True)
    
    #concat the original DataFrame with the encoded variables
    train_data_encoded = pd.concat([train_data_clean, one_hot_encoded], axis=1)
    
    #get list of one-hot encoded column names
    one_hot_encoded_list = one_hot_encoded.columns.to_list()

    return train_data_encoded, one_hot_encoded_list