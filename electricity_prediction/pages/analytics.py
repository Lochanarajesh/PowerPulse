import streamlit as st
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
