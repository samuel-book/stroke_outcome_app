"""
Streamlit app for the stroke outcome model. 
"""

# ----- Imports -----
import numpy as np
import matplotlib.pyplot as plt 
import streamlit as st

# For handling the mRS dist files: 
import pandas as pd 

# Custom functions:
from outcome_utilities.probs_with_time import \
    find_mrs_constants, find_mrs_bins_t
from outcome_utilities.plot_probs_with_time import \
    plot_probs_filled


# ----- Fixed parameters ----- 
# Define maximum treatment times:
time_no_effect_ivt = int(6.3*60)   # minutes
time_no_effect_mt = int(8*60)      # minutes

# Change default colour scheme:
plt.style.use('seaborn-colorblind')
# Get current matplotlib style sheet colours:
colour_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
# Remove extra colours:
colour_list = colour_list[:7-1]
# Add grey for mRS=6 bin:
colour_list.append('DarkSlateGray')


# ----- User-selected variables -----
occlusion_input = st.sidebar.radio(
    'Choose the occlusion type.',
    ['Non-large vessel (nLVO)',
    'Large vessel (LVO)']#,
    # 'Both nLVOs and LVOs']
    )   

treatment_input = st.sidebar.radio(
    'Choose the treatment type.',
    ['Intravenous thrombolysis (IVT)',
    'Mechanical thrombectomy (MT)']#,
    # 'Both IVT and MT']
    )

mRS_input = st.sidebar.multiselect(
    'Choose the pre-stroke mRS value(s).',
    range(6)
    )

# if any('MT' in treatment for treatment in treatment_input): 
time_no_effect = (time_no_effect_mt if 'MT' in treatment_input 
            else time_no_effect_ivt)

time_input = st.sidebar.slider(
    'Choose the treatment time in minutes.',
    0, time_no_effect, 240 # start, end, default value
    )

occlusion_str = 'nlvo' if 'nLVO' in occlusion_input else 'lvo'
treatment_str = 'mt' if 'MT' in treatment_input else 'ivt'
excess_death_str = '_'+treatment_str+'_deaths' 

title = (f'Treating {occlusion_str} with {treatment_str}' +
    f'\nat treatment time {time_input//60} hours {time_input%60} minutes')
st.write('__'+title+'__')


# ----- Select the mRS data -----
# Load mRS distributions from file: 
mrs_dists_cumsum = pd.read_csv(
    './outcome_data/mrs_dist_probs_cumsum.csv', 
    index_col='Stroke type')
mrs_dists_bins = pd.read_csv(
    './outcome_data/mrs_dist_probs_bins.csv', 
    index_col='Stroke type')

# Build the names of the dists that we need from these files
# by using the input variables: 
dist_pre_stroke_str = 'pre_stroke_' + occlusion_str
dist_no_treatment_str = 'no_treatment_' + occlusion_str
dist_t0_treatment_str = ('t0_treatment_' + occlusion_str + '_' + 
    treatment_str)
dist_no_effect_str = ('no_effect_' + occlusion_str + '_' + 
    treatment_str + '_deaths')

# Get the dists from the data array using the strings: 
dist_pre_stroke = mrs_dists_bins.loc[dist_pre_stroke_str]
dist_no_treatment = mrs_dists_bins.loc[dist_no_treatment_str]
dist_t0_treatment = mrs_dists_bins.loc[dist_t0_treatment_str]
dist_no_effect = mrs_dists_bins.loc[dist_no_effect_str]

dist_cumsum_pre_stroke = mrs_dists_cumsum.loc[dist_pre_stroke_str]
dist_cumsum_no_treatment = mrs_dists_cumsum.loc[dist_no_treatment_str]
dist_cumsum_t0_treatment = mrs_dists_cumsum.loc[dist_t0_treatment_str]
dist_cumsum_no_effect = mrs_dists_cumsum.loc[dist_no_effect_str]


# ----- Find probability with time -----
a_list, b_list, A_list = find_mrs_constants(
    dist_cumsum_t0_treatment, dist_cumsum_no_effect, time_no_effect)
# The probability P at an arbitrary treatment time is: 
# P(treatment_time) = 1.0/(1.0 + np.exp(-A -b*treatment_time)) 
# where the values of a, b, and A are different for each mRS. 


# --- Plot probability with time -----
times_to_plot = np.linspace(0, time_no_effect, 20)
probs_to_mark = []
for mRS in mRS_input:
    prob_below = dist_cumsum_pre_stroke[mRS-1] if mRS>0 else 0.0 
    prob_above = dist_cumsum_pre_stroke[mRS] if mRS<6 else 1.0 
    probs_to_mark.append(prob_below)
    probs_to_mark.append(prob_above)

fig, ax = plt.subplots()
# probs_time_plot_title = (f'Treating {occlusion_str} with {treatment_str}' +
#     f'\nat treatment time {time_input} minutes')
plot_probs_filled(A_list, b_list, times_to_plot, colour_list, 
    # probs_to_mark=np.unique(probs_to_mark), 
    ax=ax)

ax.axvline(time_input/60.0, color='w', linestyle='--')

for prob in probs_to_mark:
    ax.axhline(prob, color='w', linestyle='-')

st.pyplot(fig)
