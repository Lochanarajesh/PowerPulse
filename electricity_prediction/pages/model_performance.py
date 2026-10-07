import streamlit as st
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
