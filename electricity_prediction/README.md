# ⚡ E-PREDICT

**Electricity Consumption Prediction & Energy Intelligence Platform**

E-PREDICT is a professional Streamlit web application designed for analyzing and predicting electricity consumption patterns using Machine Learning.

## Features
- **Data Explorer:** Upload your CSV dataset or use the built-in demo data. Automatic handling of missing values and feature engineering.
- **Consumption Analytics:** Visualize consumption trends with interactive Plotly charts, heatmaps, and correlation matrices.
- **Predict Consumption:** Leverage machine learning models to forecast electricity usage based on custom parameters.
- **Model Performance:** Evaluate models (Random Forest, Gradient Boosting, Linear Regression) with R², MAE, and RMSE metrics, along with feature importance visualization.
- **Energy Insights:** Get AI-driven insights like Peak Demand Alerts and a comprehensive Energy Pattern Score.

## Technology Stack
- **Python**
- **Streamlit** (UI Framework)
- **Pandas & NumPy** (Data Processing)
- **Scikit-learn** (Machine Learning)
- **Plotly** (Interactive Visualizations)
- **Joblib** (Model Persistence)

## Installation & Setup

1. Clone or download this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Dataset Requirements
The application automatically detects the `datetime` and `target` (electricity consumption) columns. If automatic detection fails, you can manually select the columns in the Data Explorer page.

## Future Enhancements
- Integration with real-time smart meter data APIs.
- Advanced deep learning models (LSTM, Prophet) for time-series forecasting.
- Exportable PDF reports for energy audits.
