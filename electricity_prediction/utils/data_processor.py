import pandas as pd
import numpy as np

def identify_columns(df):
    date_cols = [col for col in df.columns if any(x in col.lower() for x in ['date', 'time', 'timestamp'])]
    target_cols = [col for col in df.columns if any(x in col.lower() for x in ['consumption', 'energy', 'power', 'usage', 'kwh'])]
    
    date_col = date_cols[0] if date_cols else None
    target_col = target_cols[0] if target_cols else None
    
    return date_col, target_col

def preprocess_data(df, date_col=None, target_col=None):
    df = df.copy()
    
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df = df.sort_values(by=date_col)
        
        # Feature engineering
        df['Year'] = df[date_col].dt.year
        df['Month'] = df[date_col].dt.month
        df['Day'] = df[date_col].dt.day
        df['Hour'] = df[date_col].dt.hour
        df['Day_of_Week'] = df[date_col].dt.dayofweek
        df['Is_Weekend'] = df['Day_of_Week'].isin([5, 6]).astype(int)
        
    if target_col and target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        
    # Drop completely empty columns and rows
    df = df.dropna(how='all', axis=1)
    df = df.dropna(how='all', axis=0)
    
    # Fill remaining numerical NaNs with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    return df
