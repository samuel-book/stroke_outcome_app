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
    layout='wide')

st.title('Stroke outcome modelling')
st.header('How to use this')
st.write('select stuff on the left sidebar')
# st.subheader('test')
st.write('Emoji! :ambulance: :hospital: :pill: :syringe: :hourglass_flowing_sand: :crystal_ball: :ghost: :skull: :thumbsup: :thumbsdown:' )

