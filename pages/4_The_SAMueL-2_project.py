import streamlit as st

# ----- Page setup -----
# Set page to widescreen must be first call to st.
st.set_page_config(
    page_title='Stroke outcome modelling',
    page_icon=':ambulance:',
    layout='wide')
    
st.title('SAMueL-2')
st.header('header')
st.write('We do lots of stuff and it\'s great')