import streamlit as st
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
