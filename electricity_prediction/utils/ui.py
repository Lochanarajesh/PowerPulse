import streamlit as st

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
