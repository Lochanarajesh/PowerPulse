import streamlit as st
import pandas as pd
import os
from utils.data_processor import identify_columns, preprocess_data

st.title("Data Explorer")
st.markdown("Upload your electricity consumption dataset and automatically prepare it for analysis.")

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload CSV", type="csv")

with col2:
    st.write("Or use the provided dataset:")
    if st.button("Use Household Power Consumption"):
        dataset_path = "household_power_consumption.txt"
        if os.path.exists(dataset_path):
            with st.spinner("Loading dataset..."):
                # Load a sample to keep it fast for the app, or full if preferred. Let's load full but parse dates.
                df = pd.read_csv(dataset_path, sep=';', na_values=['?'], low_memory=False)
                df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
            st.session_state.raw_df = df
            st.session_state.data_source = "Household Power Consumption"
            st.success("Dataset loaded!")
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
