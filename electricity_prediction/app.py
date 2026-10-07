import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as go
import os

from utils.ui import apply_custom_css, render_sidebar, render_footer

# Configure page
st.set_page_config(
    page_title="E-PREDICT | Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global CSS
apply_custom_css()

# Render Sidebar navigation
render_sidebar()

# Using st.navigation from newer Streamlit versions
# If not available, we could use session state and conditional rendering, but st.navigation is preferred.
# For compatibility and simplicity, let's use multipage setup if preferred, but since we are asked to have a specific structure, we'll use Streamlit's st.navigation
pages = {
    "Dashboards": [
        st.Page("pages/overview.py", title="Overview", icon="⚡"),
        st.Page("pages/data_explorer.py", title="Data Explorer", icon="🗃️"),
        st.Page("pages/analytics.py", title="Consumption Analytics", icon="📊"),
        st.Page("pages/prediction.py", title="Predict Consumption", icon="🔮"),
        st.Page("pages/model_performance.py", title="Model Performance", icon="🎯"),
        st.Page("pages/insights.py", title="Energy Insights", icon="💡"),
    ]
}

pg = st.navigation(pages)
pg.run()

# Render footer
render_footer()
