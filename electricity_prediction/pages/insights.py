import streamlit as st

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
