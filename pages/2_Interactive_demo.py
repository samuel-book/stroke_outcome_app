# -*- coding: UTF-8 -*-
"""
Streamlit app for the stroke outcome model.


TO DO 17/NOV - fix rounding to nearest even problem with printing
Printed sums (with rounding) add up to a different value than the
metric (without rounding).
"""

# ----- Imports -----
import streamlit as st

# Add an extra bit to the path if we need to.
# Try importing something as though we're running this from the same
# directory as the landing page.
try:
    from outcome_utilities.fixed_params import page_setup
except ModuleNotFoundError:
    # If the import fails, add the landing page directory to path.
    # Assume that the script is being run from the directory above
    # the landing page directory, which is called
    # stroke_outcome_app.
    import sys
    sys.path.append('./stroke_outcome_app/')

# Custom functions:
import outcome_utilities.inputs
from outcome_utilities.inputs import write_text_from_file
from outcome_utilities.plot_timeline import make_timeline_plot
from outcome_utilities.fixed_params import \
    page_setup, utility_weights, emoji_text_dict

import outcome_utilities.main_calculations
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

    # Write text at the top of the page:
    st.markdown('# Interactive demo')
    st.info(
        ':information_source: ' +
        'For acronym reference, see the introduction page.'
        )
    write_text_from_file('pages/text_for_pages/2_Intro_for_demo.txt',
                         head_lines_to_skip=2)
    # Write these in columns:
    cols_cases = st.columns(2)
    with cols_cases[0]:
        st.markdown(''.join([
            '__Case 1:__ all eligible patients ',
            'receive IVT at the IVT-only centre, ',
            'and then patients requiring MT ',
            'are transported to the IVT+MT centre ',
            'for further treatment.'
        ]))
    with cols_cases[1]:
        st.markdown(''.join([
            '__Case 2:__ all patients are transported ',
            'directly to the IVT+MT centre ',
            'and receive the appropriate treatments there.'
        ]))


    # ###########################
    # ########## SETUP ##########
    # ###########################

    # All user inputs are now in the sidebar:
    with st.sidebar:
        st.markdown('# Setup')

        # ----- Population parameters -----
        # Collect the user inputs into a dictionary of patient
        # proportions, prop_dict.
        st.markdown('## Patient population')
        # Place an empty container here now, and later
        # put the summary info in it.
        container_summary_population = st.container()

        # Proportion of nLVO / LVO / ICH:
        with st.expander('Stroke types'):
            st.warning(
                ':warning: Currently the ICH option ' +
                'does not impact the change in mRS or utility.'
                )
            prop_dict = outcome_utilities.inputs.inputs_patient_population()

        # Percentages of each group who receive treatment:
        with st.expander('Advanced options'):
            prop_dict = outcome_utilities.inputs.\
                inputs_patient_population_advanced(prop_dict)

        # Now write info in the box from earlier.
        treated_pop_perc = 100.0*prop_dict['treated_population']
        with container_summary_population:
            st.markdown(
                'Percentage of the population receiving treatment: ' +
                f'{treated_pop_perc:5.2f}%'
                )

        # ----- Timeline of patient pathway -----
        # Collect the user inputs into two main dictionaries, one for
        # case 1 and one for case 1, called case(1/2)_time_dict.
        st.markdown('## Times to treatment')
        # Place an empty container here now, and later
        # put the summary times in it.
        container_summary_times = st.container()
        # Also put the pathway plot and time inputs in here:
        with st.expander('Patient pathway'):
            container_pathway_plot = st.container()
            container_pathway_inputs = st.container()

        with container_pathway_inputs:
            # Take the inputs:
            st.write(
                'Each step uses times in minutes. ' +
                'To remove a step, set the value to zero.'
                )
            # Split the input widgets into two columns:
            cols_timeline = st.columns([1, 1])
            # Return the dictionaties and some useful times:
            (case1_time_dict, case2_time_dict, case1_time_to_ivt,
             case1_time_to_mt, case2_time_to_ivt, case2_time_to_mt) = \
                outcome_utilities.inputs.inputs_pathway(cols_timeline)

        with container_pathway_plot:
            # Draw timelines
            make_timeline_plot([case1_time_dict, case2_time_dict])

        # Now display the final times to IVT and MT
        # in the box we placed earlier:
        with container_summary_times:
            # Put Case 1 in first column, Case 2 in second.
            cols_for_times = st.columns(2)
            with cols_for_times[0]:
                st.markdown('__Case 1:__')
                st.markdown(
                    emoji_text_dict['ivt_arrival_to_treatment'] +
                    f' IVT: {case1_time_to_ivt//60}hr ' +
                    f'{case1_time_to_ivt%60}min'
                )
                st.markdown(
                    emoji_text_dict['mt_arrival_to_treatment'] +
                    f' MT: {case1_time_to_mt//60}hr ' +
                    f'{case1_time_to_mt%60}min'
                )

            with cols_for_times[1]:
                st.markdown('__Case 2:__')
                st.markdown(
                    emoji_text_dict['ivt_arrival_to_treatment'] +
                    f' IVT: {case2_time_to_ivt//60}hr ' +
                    f'{case2_time_to_ivt%60}min'
                )
                st.markdown(
                    emoji_text_dict['mt_arrival_to_treatment'] +
                    f' MT: {case2_time_to_mt//60}hr ' +
                    f'{case2_time_to_mt%60}min'
                )

    # ##################################
    # ########## CALCULATIONS ##########
    # ##################################
    # ----- Case 1 -----
    # Each of the returned dictionaries contains various useful mRS
    # distributions, lists of constants for creating probability vs
    # time, treatment time...
    nlvo_ivt_case1_dict, lvo_ivt_case1_dict, lvo_mt_case1_dict = \
        outcome_utilities.main_calculations.\
        find_dist_dicts(case1_time_to_ivt, case1_time_to_mt)

    # Make dictionary for each treatment type.
    # Each dict contains the mean mRS or utility at various times,
    # pre-stroke, no treatment, and time-input treatment,
    # and the differences between time-input treatment and the others.
    # The final "outcomes" dict gathers the expected mean mRS and
    # utility, and changes from no treatment, for the input patient
    # population.
    (mean_mRS_dict_nlvo_ivt_case1,
     mean_util_dict_nlvo_ivt_case1,
     mean_mRS_dict_lvo_ivt_case1,
     mean_util_dict_lvo_ivt_case1,
     mean_mRS_dict_lvo_mt_case1,
     mean_util_dict_lvo_mt_case1,
     mean_outcomes_dict_population_case1) = \
        outcome_utilities.main_calculations.find_outcome_dicts(
            nlvo_ivt_case1_dict,
            lvo_ivt_case1_dict,
            lvo_mt_case1_dict,
            utility_weights,
            prop_dict
            )

    # ----- Case 2 -----
    # (same as for case 1)
    nlvo_ivt_case2_dict, lvo_ivt_case2_dict, lvo_mt_case2_dict = \
        outcome_utilities.main_calculations.\
        find_dist_dicts(case2_time_to_ivt, case2_time_to_mt)

    # (same as for case 1)
    (mean_mRS_dict_nlvo_ivt_case2,
     mean_util_dict_nlvo_ivt_case2,
     mean_mRS_dict_lvo_ivt_case2,
     mean_util_dict_lvo_ivt_case2,
     mean_mRS_dict_lvo_mt_case2,
     mean_util_dict_lvo_mt_case2,
     mean_outcomes_dict_population_case2) = \
        outcome_utilities.main_calculations.find_outcome_dicts(
            nlvo_ivt_case2_dict,
            lvo_ivt_case2_dict,
            lvo_mt_case2_dict,
            utility_weights,
            prop_dict
            )

    # ###########################
    # ######### RESULTS #########
    # ###########################
    st.markdown('# Results')
    # ----- Show metric for +/- mRS and utility -----
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
    with st.expander('1: mRS distributions at the treatment times'):

        write_text_from_file('pages/text_for_pages/2_Probs_with_time.txt',
                             head_lines_to_skip=3)

        tab1, tab2, tab3, tab4 = st.tabs([
            'nLVO treated with IVT',
            'LVO treated with IVT only',
            'LVO treated with MT',
            'ICH'
        ])

        with tab1:
            outcome_utilities.container_details_prob_vs_time.\
                plot_probs_with_time(
                    nlvo_ivt_case1_dict, nlvo_ivt_case2_dict)
            outcome_utilities.container_details_prob_vs_time.\
                table_probs_with_time(
                    nlvo_ivt_case1_dict, nlvo_ivt_case2_dict)
        with tab2:
            outcome_utilities.container_details_prob_vs_time.\
                plot_probs_with_time(
                    lvo_ivt_case1_dict, lvo_ivt_case2_dict)
            outcome_utilities.container_details_prob_vs_time.\
                table_probs_with_time(
                    lvo_ivt_case1_dict, lvo_ivt_case2_dict)
        with tab3:
            outcome_utilities.container_details_prob_vs_time.\
                plot_probs_with_time(
                    lvo_mt_case1_dict, lvo_mt_case2_dict)
            outcome_utilities.container_details_prob_vs_time.\
                table_probs_with_time(
                    lvo_mt_case1_dict, lvo_mt_case2_dict)
        with tab4:
            st.write('Nothing to see here.')

    # ----- Details 2: Cumulative changes -----
    with st.expander('2: Cumulative changes in mRS and utility'):
        tab1, tab2, tab3, tab4 = st.tabs([
            'nLVO treated with IVT',
            'LVO treated with IVT only',
            'LVO treated with MT',
            'ICH'
        ])

        explanation_str = '''
            We can draw some of the data from the table in the
            "mRS distributions at the treatment times" section above
            to create bar charts of mRS probability distributions.

            The weighted mean utility and mRS is calculated using
            those regions of the chart where the mRS is different
            between the "No treatment" and "Treated at..." bars.
            Each line in the following sums calculates
            the _outcome if treated_ minus the _outcome if not treated_
            all multiplied by the _proportion of the population with
            this change in outcome_.
            '''
        with tab1:
            # nLVO IVT
            st.markdown(explanation_str)
            st.subheader('Case 1')
            outcome_utilities.container_details_cumulative_changes.\
                draw_cumulative_changes(
                    nlvo_ivt_case1_dict,
                    case1_time_to_ivt,
                    'nLVO_IVT_case1'
                    )

            st.subheader('Case 2')
            outcome_utilities.container_details_cumulative_changes.\
                draw_cumulative_changes(
                    nlvo_ivt_case2_dict,
                    case2_time_to_ivt,
                    'nLVO_IVT_case2'
                    )

        with tab2:
            # LVO IVT
            st.markdown(explanation_str)
            st.subheader('Case 1')
            outcome_utilities.container_details_cumulative_changes.\
                draw_cumulative_changes(
                    lvo_ivt_case1_dict,
                    case1_time_to_ivt,
                    'LVO_IVT_case1'
                    )

            st.subheader('Case 2')
            outcome_utilities.container_details_cumulative_changes.\
                draw_cumulative_changes(
                    lvo_ivt_case2_dict,
                    case2_time_to_ivt,
                    'LVO_IVT_case2'
                    )

        with tab3:
            # LVO MT
            st.markdown(explanation_str)
            st.subheader('Case 1')
            outcome_utilities.container_details_cumulative_changes.\
                draw_cumulative_changes(
                    lvo_mt_case1_dict,
                    case1_time_to_mt,
                    'LVO_MT_case1'
                    )

            st.subheader('Case 2')
            outcome_utilities.container_details_cumulative_changes.\
                draw_cumulative_changes(
                    lvo_mt_case2_dict,
                    case2_time_to_mt,
                    'LVO_MT_case2'
                    )

        with tab4:
            st.write('Nothing to see here.')

    # ----- Details 3: Sum up changes -----
    with st.expander('3: Calculations for overall changes in utility and mRS'):
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
