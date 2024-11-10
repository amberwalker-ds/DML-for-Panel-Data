import yaml
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def load_config(filepath):
    """Loads a YAML configuration file."""
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_to_csv(df, filepath):
    """Saves a DataFrame to a CSV file."""
    df.to_csv(filepath, index=False)

def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def create_model(model_config):
    """Dynamically creates a model from a configuration dictionary."""
    model_type = model_config['type']
    params = model_config.get('params', {})
    
    if model_type == "RandomForestRegressor":
        return RandomForestRegressor(**params)
    elif model_type == "RandomForestClassifier":
        return RandomForestClassifier(**params)
    else:
        raise ValueError(f"Model type {model_type} is not supported.")
