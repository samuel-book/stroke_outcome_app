"""
Streamlit app for the stroke outcome model.
"""

# ----- Imports -----
import streamlit as st

from outcome_utilities.inputs import write_text_from_file

# ----- Page setup -----
# Set page to widescreen must be first call to st.
st.set_page_config(
    page_title='Stroke outcome modelling',
    page_icon=':ambulance:',
    # layout='wide'
    )

write_text_from_file('pages/text_for_pages/1_Intro.txt',
                     head_lines_to_skip=4)
