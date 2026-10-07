import streamlit as st
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
