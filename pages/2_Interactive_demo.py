# -*- coding: UTF-8 -*-
"""
Streamlit app for the stroke outcome model.


TO DO 17/NOV - fix rounding to nearest even problem with printing
Printed sums (with rounding) add up to a different value than the
metric (without rounding).
"""

# ----- Imports -----
import streamlit as st
import matplotlib.pyplot as plt

# Add an extra bit to the path if we need to.
# Try importing something as though we're running this from the same
# directory as the landing page.
try:
    from outcome_utilities.fixed_params import page_setup
except ModuleNotFoundError:
    # If the import fails, add the landing page directory to path.
    # Assume that the script is being run from the directory above
    # the landing page directory, which is called
    # streamlit_lifetime_stroke.
    import sys
    sys.path.append('./stroke_outcome_app/')

# Custom functions:
from outcome_utilities.inputs import \
    inputs_patient_population, inputs_pathway, write_text_from_file, \
    inputs_patient_population_advanced
from outcome_utilities.plot_timeline import \
    make_timeline_plot
from outcome_utilities.fixed_params import \
    page_setup, colour_list, emoji_dict, utility_weights, emoji_text_dict
from outcome_utilities.main_calculations import \
    find_dist_dicts, find_outcome_dicts
# Containers:
import outcome_utilities.container_results_metrics
import outcome_utilities.container_details_cumulative_changes
import outcome_utilities.container_details_overall_changes
import outcome_utilities.container_details_prob_vs_time


def main():
    # ###########################
    # ##### START OF SCRIPT #####
    # ###########################

    page_setup()

    st.markdown('# Interactive demo')

    st.info(
        ':information_source: ' +
        'For acronym reference, see the introduction page.'
        )

    write_text_from_file('pages/text_for_pages/2_Intro_for_demo.txt',
                        head_lines_to_skip=2)


    # ###########################
    # ########## SETUP ##########
    # ###########################

    with st.sidebar:
        st.markdown('# Setup')
        # ----- Population parameters -----
        st.markdown('## Patient population')
        # Put the summary info in here later:
        container_summary_population = st.container()

        with st.expander('Stroke types'):
            st.warning(
                ':warning: Currently the ICH option ' +
                'does not impact the change in mRS or utility.'
                )
            prop_dict = inputs_patient_population()
        
        with st.expander('Advanced options'):
            prop_dict = inputs_patient_population_advanced(prop_dict)
        
        treated_pop_perc = 100.0*prop_dict['treated_population']

        # Now write info in the box from earlier.
        with container_summary_population:
            st.write('Percentage of the population receiving treatment: ',
                    f'{treated_pop_perc:5.2f}%')


        # ----- Timeline of patient pathway -----
        st.markdown('## Times to treatment')
        # Put the summary times in here later:
        container_summary_times = st.container()

        # Take the inputs:
        with st.expander('Patient pathway'):
            st.write(
                'Each step uses times in minutes. ' +
                'To remove a step, set the value to zero.'
                )
            
            cols_timeline = st.columns([1, 1])
            # with cols_timeline[0]:
            (case1_time_dict, case2_time_dict, case1_time_to_ivt,
            case1_time_to_mt, case2_time_to_ivt, case2_time_to_mt) = inputs_pathway(cols_timeline)

            # Draw timelines
            make_timeline_plot([case1_time_dict, case2_time_dict])
        
        # Now display the summary times in the box we placed earlier:
        with container_summary_times:
            # Write the final times to IVT and MT here.
            cols_for_times = st.columns(2)
            with cols_for_times[0]:
                st.markdown('__Case 1:__')
                st.markdown(
                    emoji_text_dict['ivt_arrival_to_treatment'] +
                    f' IVT: {case1_time_to_ivt//60}hr {case1_time_to_ivt%60}min'
                )
                st.markdown(
                    emoji_text_dict['mt_arrival_to_treatment'] +
                    f' IVT: {case1_time_to_mt//60}hr {case1_time_to_mt%60}min'
                )
        
            with cols_for_times[1]:
                st.markdown('__Case 2:__')
                st.markdown(
                    emoji_text_dict['ivt_arrival_to_treatment'] +
                    f' IVT: {case2_time_to_ivt//60}hr {case2_time_to_ivt%60}min'
                )
                st.markdown(
                    emoji_text_dict['mt_arrival_to_treatment'] +
                    f' IVT: {case2_time_to_mt//60}hr {case2_time_to_mt%60}min'
                )




    # ##################################
    # ########## CALCULATIONS ##########
    # ##################################
    # ----- Calculate all the stuff -----
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
    # st.subheader('Changes in mRS and utility')
    outcome_utilities.container_results_metrics.main(
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
        )


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
        outcome_utilities.container_details_prob_vs_time.main(
            nlvo_ivt_case1_dict,
            nlvo_ivt_case2_dict,
            lvo_ivt_case1_dict,
            lvo_ivt_case2_dict,
            lvo_mt_case1_dict,
            lvo_mt_case2_dict
            )


    # ----- Details 2: Cumulative changes -----
    header_expander_details_cumulative_changes = (
        '2: Cumulative changes in mRS and utility')

    with st.expander(header_expander_details_cumulative_changes):
        outcome_utilities.container_details_cumulative_changes.main(
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
            )


    # ----- Details 3: Sum up changes -----
    header_expander_details_overall_changes = (
        '3: Calculations for overall changes in utility and mRS')

    with st.expander(header_expander_details_overall_changes):
        outcome_utilities.container_details_overall_changes.main(
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
        )

if __name__ == '__main__':
    main()
