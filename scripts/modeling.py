import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score
import subprocess
import sys
# Check if 'panelsplit' is installed and install if it’s missing
try:
    from panelsplit import PanelSplit
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "panelsplit"])

def dml_process_treatment(X_base, X_lagged, Y, T, model_y, model_t, panel_split, lag):
    """
    Double Machine Learning Model (DML) for causal inference:
    - Uses cross-fitting to compute treatment effects by regressing residuals of the outcome 
      on the residuals of the treatment for unbiased theta estimates.

    Steps:
        1. Splits data using PanelSplit cross-validation.
        2. For each fold:
           - Trains a treatment model and outcome model.
           - Predicts treatment and outcome on the test set, calculating residuals.
           - Regresses outcome residuals on treatment residuals to compute theta for each fold.
        3. Averages theta values across folds and computes confidence intervals.

    Parameters:
        X_base (pd.DataFrame): DataFrame containing covariates without lags (lag_0).
        X_lagged (pd.DataFrame): DataFrame containing lagged covariates specific to treatment.
        Y (pd.Series): Target outcome variable.
        T (pd.Series): Treatment variable for causal analysis.
        model_y (sklearn estimator): Model to predict the outcome.
        model_t (sklearn estimator): Model to predict the treatment.
        panel_split (PanelSplit): PanelSplit object for cross-validation.
        lag (int): Lag index for tracking purposes.

    Returns:
        dict: A dictionary with theta estimate, confidence intervals, and performance metrics (MSE and AUC).
    """

    # Combine X_base (lag0 features) with lagged features in X_lagged for outcome model only
    X_outcome = pd.concat([X_base, X_lagged], axis=1)

    # Initialize arrays to hold residuals
    y_residuals = np.zeros_like(Y, dtype=float)
    T_residuals = np.zeros_like(T, dtype=float)
    fold_theta_estimates = []

    # Initialize lists to store metrics
    neg_mse_list = []
    auc_list = []

    # Cross-fitting using PanelSplit
    for i, (train_indices, test_indices) in enumerate(panel_split.split()):
        # For the treatment model, only use the lagged covariates (specific to the given lag)
        X_train_t, X_test_t = X_lagged.iloc[train_indices], X_lagged.iloc[test_indices]

        # For the outcome model, use both base (lag0) and lagged covariates
        X_outcome_train, X_outcome_test = X_outcome.iloc[train_indices], X_outcome.iloc[test_indices]
        Y_train, Y_test = Y.iloc[train_indices], Y.iloc[test_indices]
        T_train, T_test = T.iloc[train_indices], T.iloc[test_indices]

        # Fit models and calculate residuals as before
        model_y.fit(X_outcome_train, Y_train)
        g_predictions = model_y.predict(X_outcome_test)
        neg_mse = -mean_squared_error(Y_test, g_predictions)
        neg_mse_list.append(neg_mse)

        model_t.fit(X_train_t, T_train)
        m_predictions = model_t.predict_proba(X_test_t)
        auc = roc_auc_score(T_test, m_predictions[:, 1])
        auc_list.append(auc)

        # Compute residuals
        y_residuals[test_indices] = Y_test - g_predictions
        T_residuals[test_indices] = T_test - m_predictions[:, 1]

        # Estimate theta
        fold_theta = np.sum(T_residuals[test_indices] * y_residuals[test_indices]) / np.sum(T_residuals[test_indices] ** 2)
        fold_theta_estimates.append(fold_theta)

    # Average theta, standard error, and confidence intervals
    theta_hat = np.mean(fold_theta_estimates)
    se_theta = np.std(fold_theta_estimates) / np.sqrt(len(fold_theta_estimates))
    ci_lower = theta_hat - 1.96 * se_theta
    ci_upper = theta_hat + 1.96 * se_theta

    print(f'Lag {lag} results - Theta: {theta_hat}, CI: ({ci_lower}, {ci_upper})')

    return {
        'theta': theta_hat,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'neg_mse_avg': np.mean(neg_mse_list),
        'auc_avg': np.mean(auc_list)
    }

def prepare_and_run(df, end_period, outcome_variable, num_lags, base_covariates, ohe_list, n_splits, test_size, model_y, model_t, path_out, lag_type, X_list):
    """
    Prepares data and runs the Double Machine Learning model across specified lags:
    - Filters data based on specified end period.
    - Creates lagged covariates and treatment indicators for each lag.
    - Performs DML and saves results.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing all variables.
        end_period (int): Filter period threshold; only rows before this period are used.
        outcome_variable (str): Name of the outcome variable in df.
        num_lags (int): Number of lags for covariates and treatment.
        base_covariates (list): List of base covariates used for lagging.
        ohe_list (list): List of one-hot encoded column names for categorical variables.
        n_splits (int): Number of cross-validation splits.
        test_size (int): Number of periods in each test split for PanelSplit.
        model_y (sklearn estimator): Model to predict the outcome.
        model_t (sklearn estimator): Model to predict the treatment.
        path_out (str): Path to save DML results.
        lag_type (str): Prefix for treatment variable (e.g., 'pa' or 'any_agreement').
        X_list (list): List of features without lags for base covariates.

    Returns:
        tuple: X_lagged DataFrame for the last lag processed.
        pd.Series: Y Target outcome variable.
        pd.Series: T Treatment variable for causal analysis.
    """
    
    train_data = df[df['period'] < end_period]
    Y = train_data[outcome_variable]

    all_covariates = []

    for lag in range(0, num_lags + 1):
        lagged_covariates = [f'{cov}_lag{lag}' for cov in base_covariates]
        all_covariates.extend(lagged_covariates)
    all_covariates.extend(ohe_list)
    X = train_data[all_covariates]

    X_base = train_data[X_list]

    periods = train_data['period']
    unique_sorted_periods = pd.Series(periods.unique()).sort_values()

    panel_split = PanelSplit(
        periods,
        unique_periods=unique_sorted_periods,
        n_splits=n_splits,
        gap=0,
        test_size=test_size,
        plot=True
    )

    lag_results = {}

    for lag in range(0, num_lags + 1):
        current_lagged_covariates = [f'{cov}_lag{lag}' for cov in base_covariates] + ohe_list
        X_lagged = train_data[current_lagged_covariates] 

        T = train_data[f'{lag_type}lag{lag}']

        result = dml_process_treatment(X_base, X_lagged, Y, T, model_y, model_t, panel_split, lag)
        lag_results[lag] = result
        print(f"Lag {lag} done!")
        print(result)

    with open(path_out + 'lag_results.pkl', 'wb') as f:
        pickle.dump(lag_results, f)

    return X_lagged, T, Y