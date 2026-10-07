import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_project_structure():
    base_dir = r"c:\Users\LOCHANA\Downloads\PowerPulse\electricity_prediction"
    
    dirs = [
        "data",
        "models",
        "pages",
        "utils"
    ]
    
    for d in dirs:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)
        
    # Generate Sample Dataset
    generate_sample_data(os.path.join(base_dir, "data", "sample_electricity.csv"))
    
    # Generate Utils
    with open(os.path.join(base_dir, "utils", "__init__.py"), "w") as f: f.write("")
    
    # 1. UI Utils
    with open(os.path.join(base_dir, "utils", "ui.py"), "w", encoding="utf-8") as f:
        f.write('''import streamlit as st

def apply_custom_css():
    st.markdown("""
        <style>
        .kpi-card {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            text-align: center;
            margin-bottom: 20px;
            border: 1px solid #eef2f5;
        }
        .kpi-label {
            font-size: 0.9rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }
        .kpi-value {
            font-size: 2rem;
            font-weight: 700;
            color: #0f172a;
            margin: 0;
        }
        .kpi-unit {
            font-size: 1rem;
            color: #64748b;
            margin-left: 5px;
        }
        .stButton button {
            border-radius: 6px;
            font-weight: 600;
        }
        .stButton.primary-cta button {
            background-color: #0ea5e9;
            color: white;
            border: none;
        }
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .status-badge.low { background-color: #dcfce7; color: #166534; }
        .status-badge.normal { background-color: #e0f2fe; color: #075985; }
        .status-badge.high { background-color: #ffedd5; color: #9a3412; }
        .status-badge.critical { background-color: #fee2e2; color: #991b1b; }
        </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚡ E-PREDICT")
        st.caption("Electricity Intelligence Platform")
        
        st.divider()
        st.markdown("### Dataset Status")
        if 'df' in st.session_state and st.session_state.df is not None:
            source = st.session_state.get('data_source', 'Custom Dataset')
            st.markdown(f"● **{source}**")
        else:
            st.markdown("○ No data loaded")
            
        st.markdown("### Model")
        model_name = st.session_state.get('model_name', 'Random Forest')
        st.markdown(f"*{model_name}*")
        
        st.divider()
        
def render_footer():
    st.markdown("""
        <div style='text-align: center; margin-top: 50px; padding: 20px; color: #64748b; font-size: 0.8rem;'>
            <p>⚡ <b>E-PREDICT</b> | Electricity Consumption Prediction & Energy Intelligence</p>
            <p>Built with Python • Streamlit • Machine Learning</p>
        </div>
    """, unsafe_allow_html=True)

def render_kpi_card(label, value, unit=""):
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}<span class="kpi-unit">{unit}</span></div>
        </div>
    """, unsafe_allow_html=True)
''')

    # 2. Data Processor
    with open(os.path.join(base_dir, "utils", "data_processor.py"), "w", encoding="utf-8") as f:
        f.write('''import pandas as pd
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
''')

    # 3. Model Trainer
    with open(os.path.join(base_dir, "utils", "model_trainer.py"), "w", encoding="utf-8") as f:
        f.write('''import pandas as pd
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
''')

    # Generate Pages
    
    # 01 Overview
    with open(os.path.join(base_dir, "pages", "overview.py"), "w", encoding="utf-8") as f:
        f.write('''import streamlit as st
import pandas as pd
import plotly.express as px
from utils.ui import render_kpi_card

st.markdown("<div style='font-size: 0.8rem; font-weight: 600; color: #64748b; margin-bottom: -10px;'>ENERGY INTELLIGENCE PLATFORM</div>", unsafe_allow_html=True)
st.title("Electricity Consumption Prediction Dashboard")
st.markdown("Monitor consumption patterns, understand energy behavior, and predict future electricity demand using machine learning.")

if 'df' not in st.session_state or st.session_state.df is None:
    st.info("Please upload a dataset or load the demo dataset in the Data Explorer page.")
else:
    df = st.session_state.df
    date_col = st.session_state.get('date_col')
    target_col = st.session_state.get('target_col')
    
    if target_col and target_col in df.columns:
        # Energy Pulse feature
        latest_val = df[target_col].iloc[-1]
        p25, p75, p90 = df[target_col].quantile([0.25, 0.75, 0.90])
        
        status_class = "normal"
        status_text = "NORMAL DEMAND"
        if latest_val < p25:
            status_class = "low"
            status_text = "LOW DEMAND"
        elif latest_val > p90:
            status_class = "critical"
            status_text = "CRITICAL DEMAND"
        elif latest_val > p75:
            status_class = "high"
            status_text = "HIGH DEMAND"
            
        st.markdown(f"<div style='text-align: right; margin-top: -80px; margin-bottom: 30px;'><b>Energy Pulse:</b> <span class='status-badge {status_class}'>{status_text}</span></div>", unsafe_allow_html=True)
        
        # KPIs
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            render_kpi_card("Total Consumption", f"{df[target_col].sum():,.0f}", "kWh")
        with c2:
            render_kpi_card("Avg Consumption", f"{df[target_col].mean():,.1f}", "kWh")
        with c3:
            render_kpi_card("Peak Consumption", f"{df[target_col].max():,.1f}", "kWh")
        with c4:
            acc = st.session_state.get('model_metrics', {}).get('R2', 0)
            render_kpi_card("Model Accuracy", f"{acc:.2f}" if acc else "N/A", "R²")
        with c5:
            render_kpi_card("Last Record", f"{latest_val:,.1f}", "kWh")
            
        # Main Visualizations
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Electricity Consumption Trend")
            if date_col and date_col in df.columns:
                fig = px.line(df, x=date_col, y=target_col, template="plotly_white", line_shape="spline",
                             color_discrete_sequence=["#0ea5e9"])
                fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(df[target_col])
                
        with col2:
            st.subheader("Energy Snapshot")
            with st.container(border=True):
                st.metric("Dataset Records", f"{len(df):,}")
                st.metric("Consumption Variability (Std)", f"{df[target_col].std():,.2f}")
                if date_col:
                    st.metric("Date Range", f"{df[date_col].min().strftime('%Y-%m-%d')} to {df[date_col].max().strftime('%Y-%m-%d')}")
                    
        # Consumption Pattern
        st.subheader("Consumption Pattern")
        if 'Hour' in df.columns:
            hourly = df.groupby('Hour')[target_col].mean().reset_index()
            fig = px.bar(hourly, x='Hour', y=target_col, title="Average Hourly Consumption", template="plotly_white",
                        color=target_col, color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)
''')

    # 02 Data Explorer
    with open(os.path.join(base_dir, "pages", "data_explorer.py"), "w", encoding="utf-8") as f:
        f.write('''import streamlit as st
import pandas as pd
import os
from utils.data_processor import identify_columns, preprocess_data

st.title("Data Explorer")
st.markdown("Upload your electricity consumption dataset and automatically prepare it for analysis.")

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload CSV", type="csv")

with col2:
    st.write("Or use the demo dataset:")
    if st.button("Use Demo Dataset"):
        demo_path = "data/sample_electricity.csv"
        if os.path.exists(demo_path):
            df = pd.read_csv(demo_path)
            st.session_state.raw_df = df
            st.session_state.data_source = "Demo Dataset"
            st.success("Demo dataset loaded!")
            st.rerun()

if 'raw_df' in st.session_state:
    df_raw = st.session_state.raw_df
    
    st.success("Dataset successfully loaded")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(df_raw))
    c2.metric("Columns", len(df_raw.columns))
    c3.metric("Missing Values", df_raw.isna().sum().sum())
    c4.metric("Duplicate Rows", df_raw.duplicated().sum())
    
    date_col, target_col = identify_columns(df_raw)
    
    st.subheader("Column Configuration")
    col_x, col_y = st.columns(2)
    with col_x:
        date_col_sel = st.selectbox("Select Date/Time Column", df_raw.columns, index=df_raw.columns.get_loc(date_col) if date_col in df_raw.columns else 0)
    with col_y:
        target_col_sel = st.selectbox("Select Target Consumption Column", df_raw.columns, index=df_raw.columns.get_loc(target_col) if target_col in df_raw.columns else 0)
        
    if st.button("Process & Apply Data"):
        processed_df = preprocess_data(df_raw, date_col=date_col_sel, target_col=target_col_sel)
        st.session_state.df = processed_df
        st.session_state.date_col = date_col_sel
        st.session_state.target_col = target_col_sel
        st.success("Data processed and saved to session!")
        st.rerun()

    st.subheader("Dataset Preview")
    show_full = st.toggle("Show full dataset", value=False)
    if show_full:
        st.dataframe(df_raw)
    else:
        st.dataframe(df_raw.head(10))
        
    with st.expander("Data Quality & Preprocessing Details"):
        st.write("Missing values per column:")
        st.write(df_raw.isna().sum())
        st.write("Data types:")
        st.write(df_raw.dtypes.astype(str))
elif uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state.raw_df = df
    st.session_state.data_source = "Custom Dataset"
    st.rerun()
''')

    # 03 Analytics
    with open(os.path.join(base_dir, "pages", "analytics.py"), "w", encoding="utf-8") as f:
        f.write('''import streamlit as st
import plotly.express as px

st.title("Consumption Analytics")
st.markdown("Understand when, where, and how electricity is being consumed.")

if 'df' in st.session_state and st.session_state.df is not None:
    df = st.session_state.df
    target_col = st.session_state.target_col
    date_col = st.session_state.date_col
    
    t1, t2 = st.tabs(["Time Patterns", "Correlations"])
    
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            if 'Month' in df.columns:
                monthly = df.groupby('Month')[target_col].mean().reset_index()
                fig = px.bar(monthly, x='Month', y=target_col, title="Monthly Avg Consumption")
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            if 'Is_Weekend' in df.columns:
                weekend = df.groupby('Is_Weekend')[target_col].mean().reset_index()
                weekend['Day Type'] = weekend['Is_Weekend'].map({0: 'Weekday', 1: 'Weekend'})
                fig = px.bar(weekend, x='Day Type', y=target_col, title="Weekday vs Weekend")
                st.plotly_chart(fig, use_container_width=True)
                
        st.subheader("Energy Usage Heatmap")
        if 'Hour' in df.columns and 'Day_of_Week' in df.columns:
            heatmap_data = df.pivot_table(values=target_col, index='Day_of_Week', columns='Hour', aggfunc='mean')
            fig = px.imshow(heatmap_data, labels=dict(x="Hour of Day", y="Day of Week", color="Consumption"),
                           x=heatmap_data.columns, y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                           color_continuous_scale='Viridis', title="Average Consumption Heatmap")
            st.plotly_chart(fig, use_container_width=True)
            
    with t2:
        st.subheader("Consumption Drivers")
        num_cols = df.select_dtypes(include='number').columns.tolist()
        if len(num_cols) > 1:
            corr = df[num_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale='RdBu_r', title="Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No data available.")
''')

    # 04 Prediction
    with open(os.path.join(base_dir, "pages", "prediction.py"), "w", encoding="utf-8") as f:
        f.write('''import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.title("Predict Electricity Consumption")
st.markdown("Enter the required conditions and generate an ML-powered electricity consumption estimate.")

if 'model_features' in st.session_state and os.path.exists('models/model.pkl'):
    model = joblib.load('models/model.pkl')
    features = st.session_state.model_features
    
    with st.form("prediction_form"):
        st.subheader("Input Parameters")
        cols = st.columns(3)
        input_data = {}
        
        for i, feat in enumerate(features):
            with cols[i % 3]:
                if feat in ['Hour', 'Month', 'Day', 'Day_of_Week', 'Year']:
                    input_data[feat] = st.number_input(feat, value=int(st.session_state.df[feat].median()), step=1)
                elif feat == 'Is_Weekend':
                    input_data[feat] = st.selectbox(feat, [0, 1])
                else:
                    input_data[feat] = st.number_input(feat, value=float(st.session_state.df[feat].median()))
                    
        submit = st.form_submit_button("⚡ Predict Consumption", type="primary")
        
    if submit:
        input_df = pd.DataFrame([input_data])
        pred = model.predict(input_df)[0]
        
        st.markdown(f"""
        <div style='background-color: #f8fafc; padding: 30px; border-radius: 10px; text-align: center; border: 2px solid #0ea5e9; margin-top: 20px;'>
            <h2 style='margin:0; color:#64748b; font-size:1.2rem; font-weight:600;'>Predicted Electricity Consumption</h2>
            <h1 style='margin:10px 0; font-size:3.5rem; color:#0f172a;'>{pred:,.2f} <span style='font-size:1.5rem; color:#64748b;'>kWh</span></h1>
        </div>
        """, unsafe_allow_html=True)
else:
    st.warning("Model not trained yet. Go to Model Performance page to train a model.")
''')

    # 05 Model Performance
    with open(os.path.join(base_dir, "pages", "model_performance.py"), "w", encoding="utf-8") as f:
        f.write('''import streamlit as st
import plotly.express as px
from utils.model_trainer import train_model

st.title("Model Performance")

if 'df' in st.session_state:
    df = st.session_state.df
    target_col = st.session_state.target_col
    
    st.subheader("Select Prediction Model")
    model_name = st.selectbox("Model", ["Random Forest Regressor", "Gradient Boosting Regressor", "Linear Regression"])
    
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            features = df.select_dtypes(include='number').columns.tolist()
            features.remove(target_col)
            
            model, metrics, importance, y_test, y_pred = train_model(df, target_col, features, model_name=model_name)
            
            st.session_state.model_metrics = metrics
            st.session_state.model_features = features
            st.session_state.model_name = model_name
            st.session_state.importance = importance
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            
            st.success("✓ Model trained successfully")
            
    if 'model_metrics' in st.session_state:
        metrics = st.session_state.model_metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("R² Score", f"{metrics['R2']:.4f}")
        c2.metric("MAE", f"{metrics['MAE']:.2f}")
        c3.metric("RMSE", f"{metrics['RMSE']:.2f}")
        
        st.subheader("Actual vs Predicted")
        res_df = pd.DataFrame({'Actual': st.session_state.y_test, 'Predicted': st.session_state.y_pred})
        fig = px.scatter(res_df, x='Actual', y='Predicted', title="Prediction Accuracy", template="plotly_white")
        fig.add_shape(type="line", x0=res_df['Actual'].min(), y0=res_df['Actual'].min(), x1=res_df['Actual'].max(), y1=res_df['Actual'].max(), line=dict(dash='dash', color='red'))
        st.plotly_chart(fig, use_container_width=True)
        
        if st.session_state.importance is not None:
            st.subheader("Key Consumption Drivers (Feature Importance)")
            fig2 = px.bar(st.session_state.importance.head(10), x='Importance', y='Feature', orientation='h', title="Top Features")
            fig2.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("Load data first.")
''')

    # 06 Insights
    with open(os.path.join(base_dir, "pages", "insights.py"), "w", encoding="utf-8") as f:
        f.write('''import streamlit as st

st.title("Energy Intelligence")

if 'df' in st.session_state:
    df = st.session_state.df
    target_col = st.session_state.target_col
    
    st.subheader("Energy Pattern Score")
    # Simple score based on CoV (Coefficient of Variation)
    cov = (df[target_col].std() / df[target_col].mean()) if df[target_col].mean() != 0 else 1
    score = max(0, min(100, 100 - (cov * 100)))
    
    st.markdown(f"""
    <div style='background-color:#0f172a; color:white; padding:30px; border-radius:15px; text-align:center;'>
        <h1 style='font-size:4rem; margin:0; color:#38bdf8;'>{score:.0f}/100</h1>
        <p style='color:#94a3b8; font-size:1.1rem;'>An analytical score derived from the dataset's consumption behavior.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if 'Hour' in df.columns:
        st.subheader("Peak Hour Intelligence")
        hourly = df.groupby('Hour')[target_col].mean().sort_values(ascending=False).head(3)
        for hour, val in hourly.items():
            st.markdown(f"- **{hour:02d}:00** — {val:.1f} kWh")
            
        st.markdown("---")
        st.subheader("Smart Energy Recommendations")
        
        c1, c2 = st.columns(2)
        with c1:
            st.info("💡 **Peak Demand Alert:** Consider shifting flexible loads away from the identified peak hours to balance demand.")
        with c2:
            st.info("💡 **Consumption Stability:** Monitor periods of high variability to identify irregular equipment usage.")
else:
    st.warning("Load data to generate insights.")
''')

def generate_sample_data(filepath):
    dates = pd.date_range(start="2023-01-01", periods=1000, freq="H")
    np.random.seed(42)
    # create realistic pattern
    base = 10
    daily_pattern = np.sin(np.pi * dates.hour / 12) * 5
    noise = np.random.normal(0, 1, 1000)
    weekend_effect = np.where(dates.dayofweek >= 5, -3, 0)
    temp = 20 + np.sin(np.pi * dates.month / 6) * 10 + np.random.normal(0, 2, 1000)
    
    consumption = base + daily_pattern + noise + weekend_effect + (temp - 20) * 0.2
    consumption = np.maximum(consumption, 0)
    
    df = pd.DataFrame({
        'Datetime': dates,
        'Temperature': temp,
        'Humidity': np.random.uniform(30, 80, 1000),
        'Electricity_Consumption': consumption
    })
    df.to_csv(filepath, index=False)

if __name__ == "__main__":
    create_project_structure()
    print("Project structure created successfully!")
