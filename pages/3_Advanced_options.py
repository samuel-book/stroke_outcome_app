import streamlit as st

from outcome_utilities.inputs import write_text_from_file
from outcome_utilities.fixed_params import page_setup

page_setup() 

write_text_from_file('pages/text_for_pages/3_Advanced.txt', 
                     head_lines_to_skip=2)
