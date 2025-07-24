
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from PIL import Image
from apps.home import home_page
from apps.fraud_detection import fraud_detection_page
from apps.resources import resources_page

from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Fraud Guard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed", #(sidebar hidden)
)

pages = {
    "Home": home_page,
    "Fraud Detection": fraud_detection_page,
    "Resources": resources_page,
}

# Menu Layout 
selected_page = option_menu(
    menu_title = None,
    options=list(pages.keys()),
    icons=['house', 'shield-check', 'journal-text'],
    orientation="horizontal",
)


if selected_page in pages:
    pages[selected_page]()
else:
    st.markdown("### Invalid Page Selected")

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 14px; margin-top: 50px;'>
        🎓 Academic Project | Group 4 - Cecil Oiku, Katelyn Siu, Israel Maikyau, Meet Patel<br>
        <small>⚠️ This is a demonstration system for educational purposes only</small>
    </div>
    """,
    unsafe_allow_html=True
)