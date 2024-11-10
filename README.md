# Estmating the Causal Effect of Peace Agreements on Violence: A Double Machine Learning Approach Adapted for Panel Data

This project analyzes the causal effect of peace agreements on the intensity of violence, using **Double Machine Learning (DML)** to control for confounders and handle high-dimensional panel data. We use lagged data to capture delayed effects and fixed effects to control for individual differences between countries.

## 📈 Purpose & Features
- **Assess Causal Impact**: Use DML to better isolate the true effect of peace agreements on violence reduction.
- **Panel Data Analysis**: Incorporates lagged data and fixed effects for time-sensitive insights.
- **Custom Model Design**: Integrates machine learning models with cross-fitting for robust causal inference.

## ⚙️ Technology Stack
- **Python** with `Scikit-learn` for machine learning models
- **Pandas** and **NumPy** for data processing
- **PanelSplit** for panel data cross-validation

## 🛠️ Challenges & Future Plans
- **Challenge**: Managing high-dimensional, lagged data for accurate causal estimates.
- **Future Plans**: Add interactive visualizations, explore alternative ML models for robustness, and optimize for faster processing.

## Project Structure
```
    .
    ├── data/                             # Folder for data files (e.g., panel and PAX data)
    │   ├── panel.csv                     # Panel Data (Countries, violence levels, News topic data)
    │   ├── pax_corpus_2003_agreements_18-04-24.csv   # PA-X Data
    │   └── peace_process_to_iso.pkl      # Pkl file to help map Peace Process Names to appropriate countries
    ├── scripts/
    │   ├── data_processing.py            # Data loading and cleaning functions
    │   ├── main.py                       # Main script to run the entire process and generate estimated thetas
    │   └── modeling.py                   # Double Machine Learning (DML) process and model training
    ├── test.py                           # Test file to test the modeling and data processing files
    ├── config.yaml                       # Configuration file with project settings and paths
    ├── README.md                         # Project documentation
    ├── dml_modeling_eda_walkthrough.ipynb # Notebook to show thought process and logic when tackling the project
    └── requirements.txt                  # Required packages for the project
```


## 🚀 Quick Start
1. **Clone this repository**:
   ```bash
   git clone https://github.com/yourusername/peace-agreement-causal-impact.git

2. Set up Python Environment using requirements.txt
   ```bash
   pip install -r requirements.txt

4. Configure the project: Update config.yaml with your data paths and settings. This file includes:
   end_period, outcome_variable, num_lags, n_splits, test_size, and paths for data and results storage.


## Results and Analysis
### Key Results
- Theta Estimates: The model estimates the treatment effects (theta) with confidence intervals, providing insights into the causal impact of the treatment.
- Performance Metrics: Evaluates outcome models using metrics like Mean Squared Error (MSE) and Area Under the Curve (AUC)
### Analysis
- Treatment Effect (Theta): This is the main thing we're looking for, understanding the influence of the treatment variable on the outcome after accounting for confounders.
- Residual Analysis: The model residuals are looked at to make sure that they meet the assumptions of the DML approach.
### Outputs
- Results are saved as a .pkl file in the specified path (path_out in config.yaml), with details on each lag's theta estimates and performance metrics. With this, you can run more analysis such as sensitivy analysis with placebos (like in the ipynb file)
