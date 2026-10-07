import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import os

def get_model(name):
    if name == "Random Forest Regressor":
        return RandomForestRegressor(n_estimators=100, random_state=42)
    elif name == "Gradient Boosting Regressor":
        return GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        return LinearRegression()

def train_model(df, target_col, feature_cols, model_name="Random Forest Regressor", time_series=True):
    X = df[feature_cols]
    y = df[target_col]
    
    if time_series:
        # Chronological split
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    model = get_model(model_name)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(feature_cols, 'models/features.pkl')
    
    # Feature importance
    importance = None
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
    return model, metrics, importance, y_test, y_pred
