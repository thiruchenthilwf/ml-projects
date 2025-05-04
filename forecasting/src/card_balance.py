"""
Credit Card Balance Forecasting Model
------------------------------------
This script builds ML models to predict average household credit card balances
based on economic indicators like inflation, interest rates, tariffs, personal income,
and other relevant factors.

Requirements:
- pandas
- numpy
- scikit-learn
- matplotlib
- xgboost
- statsmodels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# 1. Data Loading and Preprocessing
def load_and_preprocess_data(file_path):
    """
    Load and preprocess the economic and credit card balance data
    """
    # Load data
    # For demonstration, we'll create synthetic data if file_path is None
    if file_path is None:
        # Create synthetic data for demonstration
        n_samples = 120  # 10 years of monthly data
        
        # Generate dates (monthly for 10 years)
        dates = pd.date_range('2013-01-01', periods=n_samples, freq='M')
        
        # Generate features with realistic seasonal patterns and trends
        np.random.seed(42)
        
        # Economic indicators with seasonal patterns and trends
        time_idx = np.arange(n_samples)
        
        # Inflation rate (%) with slight upward trend and seasonal pattern
        inflation = 2 + 0.01 * time_idx + 0.5 * np.sin(time_idx/6) + np.random.normal(0, 0.3, n_samples)
        
        # Interest rate (%) with downward trend and policy adjustments
        interest_rate = 5 - 0.02 * time_idx + 0.3 * np.sin(time_idx/12) + np.random.normal(0, 0.2, n_samples)
        interest_rate = np.maximum(0.5, interest_rate)  # Keep rates above 0.5%
        
        # Tariff rates (%) with step changes
        tariff = 2 + 0.5 * (time_idx > 60) + 1 * (time_idx > 80) + np.random.normal(0, 0.1, n_samples)
        
        # Personal income ($1000s) with upward trend and seasonal bonuses
        personal_income = 50 + 0.1 * time_idx + 2 * np.sin(time_idx/12 + 11) + np.random.normal(0, 1, n_samples)
        
        # Unemployment rate (%) with cyclical pattern
        unemployment = 6 - 0.02 * time_idx + 1 * np.sin(time_idx/24) + np.random.normal(0, 0.3, n_samples)
        unemployment = np.maximum(3, unemployment)  # Keep unemployment above 3%
        
        # Consumer confidence index (0-100)
        consumer_confidence = 70 + 0.05 * time_idx - 5 * (time_idx > 70) + 0.7 * np.sin(time_idx/12) + np.random.normal(0, 3, n_samples)
        
        # Housing prices index
        housing_price_idx = 200 + 1 * time_idx + 5 * np.sin(time_idx/12) + np.random.normal(0, 3, n_samples)
        
        # Stock market performance (index points)
        stock_market = 1000 + 5 * time_idx + 50 * np.sin(time_idx/12) + np.random.cumsum(np.random.normal(0, 20, n_samples))
        
        # GDP growth rate (%)
        gdp_growth = 2.5 + 0.5 * np.sin(time_idx/12) + np.random.normal(0, 0.5, n_samples)
        
        # Retail sales growth (%)
        retail_sales = 3 + 2 * np.sin(time_idx/12 + 6) + np.random.normal(0, 1, n_samples)
        
        # Savings rate (%)
        savings_rate = 8 - 0.01 * time_idx + 0.5 * np.sin(time_idx/12) + np.random.normal(0, 0.5, n_samples)
        
        # Generate target: Average credit card balance ($)
        # Complex function of economic factors
        cc_balance = (
            5000  # Base amount
            + 20 * time_idx  # Upward trend
            + 300 * np.sin(time_idx/12 + 3)  # Seasonal pattern (holiday spending)
            + 500 * (inflation - 2)  # Inflation effect
            - 300 * (interest_rate - 3)  # Interest rate effect (negative)
            + 100 * tariff  # Tariff effect
            + 0.1 * personal_income  # Income effect
            + 100 * unemployment  # Unemployment effect
            - 20 * consumer_confidence  # Consumer confidence (negative)
            + 0.5 * (housing_price_idx - 200)  # Housing price effect
            + 0.1 * (stock_market/100)  # Stock market effect
            + 100 * gdp_growth  # GDP growth effect
            + 50 * retail_sales  # Retail sales effect
            - 100 * savings_rate  # Savings rate effect (negative)
            + np.random.normal(0, 200, n_samples)  # Random noise
        )
        
        # Create DataFrame
        data = pd.DataFrame({
            'date': dates,
            'inflation': inflation,
            'interest_rate': interest_rate,
            'tariff': tariff,
            'personal_income': personal_income,
            'unemployment': unemployment,
            'consumer_confidence': consumer_confidence,
            'housing_price_idx': housing_price_idx,
            'stock_market': stock_market,
            'gdp_growth': gdp_growth,
            'retail_sales': retail_sales,
            'savings_rate': savings_rate,
            'cc_balance': cc_balance
        })
        
        # Add month and quarter as categorical features
        data['month'] = data['date'].dt.month
        data['quarter'] = data['date'].dt.quarter
        data['year'] = data['date'].dt.year
        
        # Add lag features of important variables
        for lag in [1, 3, 6, 12]:
            data[f'cc_balance_lag{lag}'] = data['cc_balance'].shift(lag)
            data[f'interest_rate_lag{lag}'] = data['interest_rate'].shift(lag)
            data[f'inflation_lag{lag}'] = data['inflation'].shift(lag)
            data[f'personal_income_lag{lag}'] = data['personal_income'].shift(lag)
        
        # Drop rows with NaN (due to lag creation)
        data = data.dropna()
    else:
        # Load actual data from file
        data = pd.read_csv(file_path, parse_dates=['date'])
        
        # Perform data cleaning steps here
        # data = data.dropna()  # Or use imputation
        
        # Feature engineering
        data['month'] = data['date'].dt.month
        data['quarter'] = data['date'].dt.quarter
        data['year'] = data['date'].dt.year
        
        # Add lag features based on domain knowledge
        for lag in [1, 3, 6, 12]:
            data[f'cc_balance_lag{lag}'] = data['cc_balance'].shift(lag)
            
        # If these columns exist
        if 'interest_rate' in data.columns:
            for lag in [1, 3, 6]:
                data[f'interest_rate_lag{lag}'] = data['interest_rate'].shift(lag)
                
        if 'inflation' in data.columns:
            for lag in [1, 3, 6]:
                data[f'inflation_lag{lag}'] = data['inflation'].shift(lag)
                
        if 'personal_income' in data.columns:
            for lag in [1, 3, 6]:
                data[f'personal_income_lag{lag}'] = data['personal_income'].shift(lag)
        
        # Drop rows with NaN values
        data = data.dropna()
    
    # Sort by date
    data = data.sort_values('date')
    
    return data

# 2. Feature Selection and Engineering
def prepare_features(data):
    """
    Prepare features for model training
    """
    # Define feature columns
    numeric_features = [
        'inflation', 'interest_rate', 'tariff', 'personal_income',
        'unemployment', 'consumer_confidence', 'housing_price_idx',
        'stock_market', 'gdp_growth', 'retail_sales', 'savings_rate',
        'cc_balance_lag1', 'cc_balance_lag3', 'cc_balance_lag6', 'cc_balance_lag12',
        'interest_rate_lag1', 'interest_rate_lag3', 'interest_rate_lag6',
        'inflation_lag1', 'inflation_lag3', 'inflation_lag6',
        'personal_income_lag1', 'personal_income_lag3', 'personal_income_lag6'
    ]
    
    # Filter to only include columns that exist in the dataframe
    numeric_features = [col for col in numeric_features if col in data.columns]
    
    categorical_features = ['month', 'quarter']
    
    # Extract features and target
    X = data[numeric_features + categorical_features].copy()
    y = data['cc_balance'].copy()
    
    # Keep date for time series analysis
    dates = data['date']
    
    return X, y, dates, numeric_features, categorical_features

# 3. Model Building Functions
def build_and_evaluate_models(X, y, numeric_features, categorical_features):
    """
    Build and evaluate multiple regression models
    """
    # Split data into training and testing sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # No shuffle for time series data
    )
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])
    
    # Define models to evaluate
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42)
    }
    
    # Store results
    results = {}
    trained_models = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        # Create pipeline with preprocessing
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R^2': r2
        }
        
        trained_models[name] = pipeline
    
    return results, trained_models, X_train, X_test, y_train, y_test

# 4. Advanced Model Tuning
def tune_best_model(X_train, y_train, X_test, y_test, numeric_features, categorical_features):
    """
    Perform hyperparameter tuning for XGBoost model
    """
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])
    
    # Apply preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Define parameter grid for XGBoost
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Use time series split for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Create and tune model
    xgb = XGBRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X_train_processed, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test_processed)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best model RMSE: {rmse:.2f}")
    print(f"Best model R^2: {r2:.4f}")
    
    # Create full pipeline with best model
    best_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', best_model)
    ])
    
    return best_pipeline, grid_search.best_params_

# 5. Time Series Analysis using SARIMA
def build_sarima_model(data):
    """
    Build a SARIMA model for time series forecasting
    """
    # Extract the time series
    ts = data.set_index('date')['cc_balance']
    
    # Train-test split
    train_size = int(len(ts) * 0.8)
    train, test = ts[:train_size], ts[train_size:]
    
    # Fit SARIMA model
    # Order: (p,d,q) and seasonal order: (P,D,Q,s)
    # p,P: AR terms, d,D: differencing, q,Q: MA terms, s: seasonality
    model = SARIMAX(
        train,
        order=(1, 1, 1),              # Non-seasonal components
        seasonal_order=(1, 1, 1, 12),  # Seasonal components (monthly)
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    results = model.fit(disp=False)
    
    # Make predictions
    forecast = results.get_forecast(steps=len(test))
    forecast_ci = forecast.conf_int()
    
    # Evaluate
    y_pred = forecast.predicted_mean
    rmse = np.sqrt(mean_squared_error(test, y_pred))
    
    print(f"SARIMA RMSE: {rmse:.2f}")
    
    return results, train, test, y_pred, forecast_ci

# 6. Feature Importance Analysis
def analyze_feature_importance(best_model, feature_names):
    """
    Analyze and visualize feature importance
    """
    # Check if the model has feature_importances_ attribute
    if hasattr(best_model, 'feature_importances_'):
        # Get feature importances
        importances = best_model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()
        
        # Return sorted features and importances
        return [(feature_names[i], importances[i]) for i in indices]
    else:
        print("Model doesn't provide feature importances")
        return None

# 7. Forecasting Future Credit Card Balances
def forecast_future(best_model, last_data, periods=24, future_economic_scenarios=None):
    """
    Forecast future credit card balances based on different economic scenarios
    
    Parameters:
    - best_model: Trained model
    - last_data: Last known data points
    - periods: Number of months to forecast
    - future_economic_scenarios: Dict of scenarios with economic projections
    """
    if future_economic_scenarios is None:
        # Default economic scenarios if none provided
        future_economic_scenarios = {
            'Baseline': {
                'inflation': last_data['inflation'].iloc[-1],
                'interest_rate': last_data['interest_rate'].iloc[-1],
                'tariff': last_data['tariff'].iloc[-1],
                'personal_income': last_data['personal_income'].iloc[-1],
                'unemployment': last_data['unemployment'].iloc[-1],
                'consumer_confidence': last_data['consumer_confidence'].iloc[-1],
                'housing_price_idx': last_data['housing_price_idx'].iloc[-1],
                'stock_market': last_data['stock_market'].iloc[-1],
                'gdp_growth': last_data['gdp_growth'].iloc[-1],
                'retail_sales': last_data['retail_sales'].iloc[-1],
                'savings_rate': last_data['savings_rate'].iloc[-1]
            },
            'Economic Growth': {
                'inflation': last_data['inflation'].iloc[-1] + 0.5,
                'interest_rate': last_data['interest_rate'].iloc[-1] + 0.25,
                'tariff': last_data['tariff'].iloc[-1] - 0.5,
                'personal_income': last_data['personal_income'].iloc[-1] * 1.02,
                'unemployment': max(2.0, last_data['unemployment'].iloc[-1] - 0.5),
                'consumer_confidence': min(100, last_data['consumer_confidence'].iloc[-1] + 5),
                'housing_price_idx': last_data['housing_price_idx'].iloc[-1] * 1.03,
                'stock_market': last_data['stock_market'].iloc[-1] * 1.05,
                'gdp_growth': last_data['gdp_growth'].iloc[-1] + 0.8,
                'retail_sales': last_data['retail_sales'].iloc[-1] + 1.0,
                'savings_rate': last_data['savings_rate'].iloc[-1] - 0.2
            },
            'Economic Recession': {
                'inflation': max(0, last_data['inflation'].iloc[-1] - 0.5),
                'interest_rate': max(0, last_data['interest_rate'].iloc[-1] - 1.0),
                'tariff': last_data['tariff'].iloc[-1] + 1.0,
                'personal_income': last_data['personal_income'].iloc[-1] * 0.98,
                'unemployment': last_data['unemployment'].iloc[-1] + 1.5,
                'consumer_confidence': max(30, last_data['consumer_confidence'].iloc[-1] - 15),
                'housing_price_idx': last_data['housing_price_idx'].iloc[-1] * 0.95,
                'stock_market': last_data['stock_market'].iloc[-1] * 0.85,
                'gdp_growth': last_data['gdp_growth'].iloc[-1] - 2.0,
                'retail_sales': last_data['retail_sales'].iloc[-1] - 2.0,
                'savings_rate': last_data['savings_rate'].iloc[-1] + 1.0
            }
        }
    
    # Create forecasts for each scenario
    forecasts = {}
    last_date = last_data['date'].iloc[-1]
    
    for scenario_name, scenario in future_economic_scenarios.items():
        # Create future dates
        future_dates = pd.date_range(start=pd.Timestamp(last_date) + pd.DateOffset(months=1), 
                                    periods=periods, freq='M')
        
        # Initialize forecast dataframe
        forecast_df = pd.DataFrame(index=range(periods))
        forecast_df['date'] = future_dates
        
        # Start with last known values
        for col in last_data.columns:
            if col != 'date' and col != 'cc_balance':
                forecast_df[col] = last_data[col].iloc[-1]
        
        # Apply scenario adjustments for each month
        for i in range(periods):
            month = i % 12 + 1
            forecast_df.loc[i, 'month'] = month
            forecast_df.loc[i, 'quarter'] = (month - 1) // 3 + 1
            forecast_df.loc[i, 'year'] = future_dates[i].year
            
            # Linear change factors for scenario over time
            factor = (i + 1) / periods
            
            # Update economic indicators based on scenario
            for indicator, target_value in scenario.items():
                if indicator in forecast_df.columns:
                    current = last_data[indicator].iloc[-1]
                    change = target_value - current
                    forecast_df.loc[i, indicator] = current + change * factor
        
        # Initialize with last known CC balance
        last_balance = last_data['cc_balance'].iloc[-1]
        
        # Forecast month by month
        predicted_balances = []
        
        # Copy to avoid modifying the original
        forecast_working = forecast_df.copy()
        
        for i in range(periods):
            # For the first prediction, use last known values for lagged features
            if i == 0:
                forecast_working.loc[i, 'cc_balance_lag1'] = last_data['cc_balance'].iloc[-1]
                forecast_working.loc[i, 'cc_balance_lag3'] = last_data['cc_balance'].iloc[-3] if len(last_data) >= 3 else last_data['cc_balance'].iloc[-1]
                forecast_working.loc[i, 'cc_balance_lag6'] = last_data['cc_balance'].iloc[-6] if len(last_data) >= 6 else last_data['cc_balance'].iloc[-1]
                forecast_working.loc[i, 'cc_balance_lag12'] = last_data['cc_balance'].iloc[-12] if len(last_data) >= 12 else last_data['cc_balance'].iloc[-1]
                
                # Other lagged variables if they exist
                for var in ['interest_rate', 'inflation', 'personal_income']:
                    for lag in [1, 3, 6]:
                        lag_col = f'{var}_lag{lag}'
                        if lag_col in forecast_working.columns:
                            lag_idx = min(lag, len(last_data) - 1)
                            forecast_working.loc[i, lag_col] = last_data[var].iloc[-lag_idx]
            else:
                # Update lagged values based on previous predictions
                forecast_working.loc[i, 'cc_balance_lag1'] = predicted_balances[-1]
                forecast_working.loc[i, 'cc_balance_lag3'] = predicted_balances[-3] if i >= 3 else last_data['cc_balance'].iloc[-1]
                forecast_working.loc[i, 'cc_balance_lag6'] = predicted_balances[-6] if i >= 6 else last_data['cc_balance'].iloc[-1]
                forecast_working.loc[i, 'cc_balance_lag12'] = predicted_balances[-12] if i >= 12 else last_data['cc_balance'].iloc[-1]
                
                # Update other lagged variables
                for var in ['interest_rate', 'inflation', 'personal_income']:
                    for lag in [1, 3, 6]:
                        lag_col = f'{var}_lag{lag}'
                        if lag_col in forecast_working.columns:
                            if i >= lag:
                                forecast_working.loc[i, lag_col] = forecast_working.loc[i-lag, var]
                            else:
                                lag_idx = min(lag, len(last_data) - 1) 
                                forecast_working.loc[i, lag_col] = last_data[var].iloc[-lag_idx]
            
            # Make prediction for this month
            feature_row = forecast_working.iloc[[i]]
            prediction = best_model.predict(feature_row.drop(columns=['date']))
            predicted_balances.append(prediction[0])
        
        # Add predictions to forecast dataframe
        forecast_df['predicted_cc_balance'] = predicted_balances
        forecasts[scenario_name] = forecast_df
    
    return forecasts

# 8. Visualization Functions
def plot_model_comparison(results):
    """
    Plot model comparison results
    """
    # Extract metrics
    models = list(results.keys())
    rmse_values = [results[model]['RMSE'] for model in models]
    r2_values = [results[model]['R^2'] for model in models]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot RMSE (lower is better)
    ax1.barh(models, rmse_values, color='skyblue')
    ax1.set_title('RMSE by Model (Lower is Better)')
    ax1.set_xlabel('RMSE')
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Plot R² (higher is better)
    ax2.barh(models, r2_values, color='lightgreen')
    ax2.set_title('R² by Model (Higher is Better)')
    ax2.set_xlabel('R²')
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def plot_actual_vs_predicted(y_test, y_pred, dates_test):
    """
    Plot actual vs predicted values
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates_test, y_test, label='Actual', color='blue')
    plt.plot(dates_test, y_pred, label='Predicted', color='red', linestyle='--')
    plt.title('Actual vs Predicted Credit Card Balances')
    plt.xlabel('Date')
    plt.ylabel('Average Credit Card Balance ($)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_sarima_forecast(train, test, forecast, dates, forecast_ci):
    """
    Plot SARIMA forecast with confidence intervals
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates[:len(train)], train, label='Training Data')
    plt.plot(dates[len(train):len(train)+len(test)], test, label='Actual Test Data')
    plt.plot(dates[len(train):len(train)+len(test)], forecast, color='red', label='SARIMA Forecast')
    
    # Plot confidence intervals
    plt.fill_between(
        dates[len(train):len(train)+len(test)],
        forecast_ci.iloc[:, 0], 
        forecast_ci.iloc[:, 1], 
        color='pink', alpha=0.3
    )
    
    plt.title('SARIMA Forecast with 95% Confidence Interval')
    plt.xlabel('Date')
    plt.ylabel('Average Credit Card Balance ($)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_scenario_forecasts(forecasts, historical_data):
    """
    Plot forecasts under different economic scenarios
    """
    plt.figure(figsize=(14, 7))
    
    # Plot historical data
    plt.plot(
        historical_data['date'], 
        historical_data['cc_balance'],
        color='black',
        label='Historical Data'
    )
    
    # Plot each scenario
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    linestyles = ['-', '--', '-.', ':', '-']
    
    for i, (scenario_name, forecast_df) in enumerate(forecasts.items()):
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        
        plt.plot(
            forecast_df['date'],
            forecast_df['predicted_cc_balance'],
            color=color,
            linestyle=linestyle,
            label=f'Forecast: {scenario_name}'
        )
    
    plt.title('Credit Card Balance Forecasts Under Different Economic Scenarios')
    plt.xlabel('Date')
    plt.ylabel('Average Credit Card Balance ($)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# 9. Main Function
def main(file_path=None):
    """
    Main function to run the entire analysis
    """
    print("Loading and preprocessing data...")
    data = load_and_preprocess_data(file_path)
    print(f"Data shape: {data.shape}")
    
    print("\nPreparing features...")
    X, y, dates, numeric_features, categorical_features = prepare_features(data)
    
    print("\nBuilding and evaluating models...")
    results, trained_models, X_train, X_test, y_train, y_test = build_and_evaluate_models(
        X, y, numeric_features, categorical_features
    )
if __name__ == "__main__":
    main()
