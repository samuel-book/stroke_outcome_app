"""
Streamlit app for the stroke outcome model. 
"""

# ----- Imports -----
import streamlit as st

# ----- Page setup -----
# Set page to widescreen must be first call to st.
st.set_page_config(
    page_title='Stroke outcome modelling',
    page_icon=':ambulance:',
    # layout='wide'
    )

st.title('Stroke outcome modelling')

st.header('How to use this')
st.write('Go to the interactive demo in the left sidebar')

st.header('modified Rankin Scale (mRS) and utility')
st.write('To do - add descriptions')

