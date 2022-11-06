# -*- coding: UTF-8 -*-
"""
Streamlit app for the stroke outcome model. 


TO DO 06/NOV - fix rounding to nearest even problem with printing
(sums currently don't add up when the preferred precision is printed)
"""

# ----- Imports -----
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt 

# Custom functions:
from outcome_utilities.inputs import \
    inputs_patient_population, inputs_pathway
from outcome_utilities.plot_timeline import \
    plot_timeline, make_timeline_plot
from outcome_utilities.fixed_params import \
    colour_list, make_fig_legend, time_no_effect_ivt, time_no_effect_mt, \
    emoji_dict, utility_weights




# ###########################
# ##### START OF SCRIPT #####
# ###########################


# ----- Page setup -----
# Set page to widescreen must be first call to st.
st.set_page_config(
    page_title='Stroke outcome modelling',
    page_icon=':ambulance:',
    # layout='wide'
    )

st.title('Stroke outcome modelling')

# Define the two cases: 
st.write('We can compare the expected outcomes for two cases. ')
st.write('__Case 1:__ all eligible patients receive IVT at the IVT-only centre, and then patients requiring MT are transported to the IVT+MT centre for further treatment.')
st.write('__Case 2:__ all patients are transported directly to the IVT+MT centre and receive the appropriate treatments there.')


# ###########################
# ########## SETUP ##########
# ###########################
st.header('Setup')
# ----- Population parameters -----
st.subheader('Patient population')
st.warning(':warning: Currently the ICH option does not impact the change in mRS or utility.')
prop_dict = inputs_patient_population()


# ----- Timeline of patient pathway -----
st.subheader('Patient pathway')
st.write('Each step uses times in minutes. To remove a step, set the value to zero.')
(case1_time_dict, case2_time_dict, case1_time_to_ivt, 
 case1_time_to_mt, case2_time_to_ivt, case2_time_to_mt) = inputs_pathway()

# Draw timelines 
fig, ax = plt.subplots(figsize=(12,8))
make_timeline_plot(ax, [case1_time_dict, case2_time_dict], emoji_dict)
st.pyplot(fig) 



# ----- Calculate all the stuff ----- 
from outcome_utilities.main_calculations import \
    find_dist_dicts, find_outcome_dicts
# Case 1: 
nlvo_ivt_case1_dict, lvo_ivt_case1_dict, lvo_mt_case1_dict = \
    find_dist_dicts(case1_time_to_ivt, case1_time_to_mt)


(mean_mRS_dict_nlvo_ivt_case1, 
 mean_util_dict_nlvo_ivt_case1, 
 mean_mRS_dict_lvo_ivt_case1, 
 mean_util_dict_lvo_ivt_case1, 
 mean_mRS_dict_lvo_mt_case1, 
 mean_util_dict_lvo_mt_case1, 
 mean_outcomes_dict_population_case1) = \
    find_outcome_dicts(
        nlvo_ivt_case1_dict,
        lvo_ivt_case1_dict,
        lvo_mt_case1_dict,
        utility_weights,
        prop_dict
        )



# Case 2:
nlvo_ivt_case2_dict, lvo_ivt_case2_dict, lvo_mt_case2_dict = \
    find_dist_dicts(case2_time_to_ivt, case2_time_to_mt)
# ich_case2_dict = find_useful_dist_dict(occlusion_input, treatment_input, time_input)


(mean_mRS_dict_nlvo_ivt_case2, 
 mean_util_dict_nlvo_ivt_case2, 
 mean_mRS_dict_lvo_ivt_case2, 
 mean_util_dict_lvo_ivt_case2, 
 mean_mRS_dict_lvo_mt_case2, 
 mean_util_dict_lvo_mt_case2, 
 mean_outcomes_dict_population_case2) = \
    find_outcome_dicts(
        nlvo_ivt_case2_dict,
        lvo_ivt_case2_dict,
        lvo_mt_case2_dict,
        utility_weights,
        prop_dict
        )



# ###########################
# ######### RESULTS #########
# ###########################
st.header('Results')
# ----- Show metric for +/- mRS and utility -----
st.subheader('Changes in mRS and utility')
import outcome_utilities.container_results_metrics
outcome_utilities.container_results_metrics.main([
    # Case 1:
    mean_outcomes_dict_population_case1['mRS_treated'],
    mean_outcomes_dict_population_case1['mRS_change'],
    mean_outcomes_dict_population_case1['util_treated'],
    mean_outcomes_dict_population_case1['util_change'],
    case1_time_to_ivt,
    case1_time_to_mt,
    # Case 2:
    mean_outcomes_dict_population_case2['mRS_treated'],
    mean_outcomes_dict_population_case2['mRS_change'],
    mean_outcomes_dict_population_case2['util_treated'],
    mean_outcomes_dict_population_case2['util_change'],
    case2_time_to_ivt,
    case2_time_to_mt,
    ])



# ###########################
# ######### DETAILS #########
# ###########################
st.write('-'*50)
st.header('Details of the calculation')
st.write('The following bits detail the calculation.')


# ----- Details 1: Probability vs time ----- 
header_expander_details_prob_vs_time = (
    '1: mRS distributions at the treatment times')

with st.expander(header_expander_details_prob_vs_time): 
    import outcome_utilities.container_details_prob_vs_time
    outcome_utilities.container_details_prob_vs_time.main([
        nlvo_ivt_case1_dict,
        nlvo_ivt_case2_dict,
        lvo_ivt_case1_dict,
        lvo_ivt_case2_dict,
        lvo_mt_case1_dict,
        lvo_mt_case2_dict
        ])


# ----- Details 2: Cumulative changes ----- 
header_expander_details_cumulative_changes = (
    '2: Cumulative changes in mRS and utility')

with st.expander(header_expander_details_cumulative_changes): 
    import outcome_utilities.container_details_cumulative_changes 
    outcome_utilities.container_details_cumulative_changes.main([
        colour_list, 
        nlvo_ivt_case1_dict,  
        nlvo_ivt_case2_dict,
        lvo_ivt_case1_dict,  
        lvo_ivt_case2_dict,
        lvo_mt_case1_dict,  
        lvo_mt_case2_dict,
        case1_time_to_ivt,
        case2_time_to_ivt,
        case1_time_to_mt,
        case2_time_to_mt
        ])


# ----- Details 3: Sum up changes ----- 
header_expander_details_overall_changes = (
    '3: Calculations for overall changes in utility and mRS')

with st.expander(header_expander_details_overall_changes): 
    import outcome_utilities.container_details_overall_changes 
    outcome_utilities.container_details_overall_changes.main([
        prop_dict, 
        # 
        mean_mRS_dict_nlvo_ivt_case1,
        mean_mRS_dict_lvo_ivt_case1,
        mean_mRS_dict_lvo_mt_case1,
        mean_outcomes_dict_population_case1['mRS_change'],
        # 
        mean_mRS_dict_nlvo_ivt_case2,
        mean_mRS_dict_lvo_ivt_case2,
        mean_mRS_dict_lvo_mt_case2,
        mean_outcomes_dict_population_case2['mRS_change'],
        # 
        mean_util_dict_nlvo_ivt_case1,
        mean_util_dict_lvo_ivt_case1,
        mean_util_dict_lvo_mt_case1,
        mean_outcomes_dict_population_case1['util_change'],
        # 
        mean_util_dict_nlvo_ivt_case2,
        mean_util_dict_lvo_ivt_case2,
        mean_util_dict_lvo_mt_case2,
        mean_outcomes_dict_population_case2['util_change'],
    ])

