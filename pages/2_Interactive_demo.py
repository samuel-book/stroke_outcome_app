# -*- coding: UTF-8 -*-
"""
Streamlit app for the stroke outcome model. 
"""

# ----- Imports -----
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt 

# For handling the mRS dist files: 
import pandas as pd 
# # For highlighting text:
# import matplotlib.patheffects as path_effects

# Custom functions:
from outcome_utilities.probs_with_time import \
    find_mrs_constants, find_mrs_bins_t
from outcome_utilities.plot_probs_with_time import \
    plot_probs_filled
from outcome_utilities.plot_dist import \
    draw_horizontal_bar, draw_connections
from outcome_utilities.bar_sizes import \
    find_bin_size_ratios
from outcome_utilities.added_utility_between_dists import \
    find_added_utility_between_dists, calculate_mean_changes, \
    calculate_combo_mean_changes, find_weighted_change
from outcome_utilities.inputs import \
    find_useful_dist_dict, inputs_patient_population, inputs_pathway, \
    utility_weights
from outcome_utilities.plot_timeline import \
    plot_timeline, plot_emoji_on_timeline, make_timeline_plot
from outcome_utilities.make_dataframe import make_combo_mRS_bin_dataframe
from outcome_utilities.change_sums import \
    do_change_sums

# ----- Functions ----- 
def make_colour_list():
    """Define the colours for plotting mRS bins."""
    # Get current matplotlib style sheet colours:
    colour_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Remove extra colours:
    colour_list = colour_list[:6]
    # Add grey for mRS=6 bin:
    colour_list.append('DarkSlateGray')
    # Change to array for easier indexing later:
    colour_list = np.array(colour_list)
    return colour_list


def make_fig_legend(colour_list):
    """Plot a legend for the mRS colours."""
    fig_legend = plt.figure(figsize=(6,2))
    # Dummy data for legend:
    dummies = []
    for i in range(7):
        dummy = plt.bar(np.NaN, np.NaN, color=colour_list[i], edgecolor='k')
        dummies.append(dummy)

    # Clear to remove automatic blank axis: 
    fig_legend.clear()
    # Draw legend using dummy bars: 
    fig_legend.legend([*dummies], range(7), 
                loc='center',ncol=7, title='mRS colour scheme', 
                ) 
    return fig_legend


def plot_utility_chart(ax_util_change, mRS_dist_mix, weighted_added_utils):

    ax_util_change.plot(mRS_dist_mix, weighted_added_utils, color='k', 
        label='Cumulative weighted added utility')

    ax_util_change.set_ylabel('Cumulative\nweighted added utility')
    ax_util_change.set_xlabel('Probability')
    ax_util_change.tick_params(top=True, right=True, which='both')

    # ax_util_change.set_ylim(ylim_util_change)
    ylim_util_change = list(ax_util_change.get_ylim())
    y_span = ylim_util_change[1]-ylim_util_change[0]
    ylim_util_change[0] -= 0.05 * y_span 
    ylim_util_change[1] += 0.05 * y_span
    ax_util_change.set_ylim(ylim_util_change)
    return ylim_util_change
    
    
def plot_bars(dists_to_bar, dists_cumsum_to_bar, ax_bars, 
    time_input, y_list, bar_height):
    y_labels = [
        'Pre-stroke', 
        (f'Treated at '+'\n'
        + f'{time_input//60}hr '
        + f'{time_input%60:02d}min'),
        'No treatment'
        ]
    # ^ keep formatting for e.g. 01 minutes in the middle bar 
    # otherwise the axis jumps about as the label changes size
    # between "9 minutes" and "10 minutes" (extra character). 

    for i, dist in enumerate(dists_to_bar):
        draw_horizontal_bar(dist, y=y_list[i], 
            colour_list=colour_list, bar_height=0.5,
            ax=ax_bars)

    ax_bars.set_yticks(y_list)
    ax_bars.set_yticklabels(y_labels)

    # Remove sides of the frame:
    for spine in ['left', 'right', 'top']:
        ax_bars.spines[spine].set_color(None)


def draw_mRS_colours_on_utility_chart(mRS_dist_mix, weighted_added_utils, 
    mRS_list_time_input_treatment, mRS_list_no_treatment, ylim_util_change, ax_util_change):
    for i in range(1,len(mRS_dist_mix)):
        y_offset = 0.05 * (ylim_util_change[1]-ylim_util_change[0])
        ax_util_change.fill_between(
            [mRS_dist_mix[i-1], mRS_dist_mix[i]], 
            np.array([weighted_added_utils[i-1], weighted_added_utils[i]])+y_offset, 
            np.array([weighted_added_utils[i-1], weighted_added_utils[i]]), 
            color=colour_list[mRS_list_time_input_treatment[i]],
            edgecolor='None',
            # zorder=0,
            )


        ax_util_change.fill_between(
            [mRS_dist_mix[i-1], mRS_dist_mix[i]], 
            np.array([weighted_added_utils[i-1], weighted_added_utils[i]])-y_offset, 
            np.array([weighted_added_utils[i-1], weighted_added_utils[i]]), 
            color=colour_list[mRS_list_no_treatment[i]],
            edgecolor='None',
            # zorder=0,
            )
    ax_util_change.set_ylim(ylim_util_change)



def do_probs_with_time(
        time_no_effect, A_list, b_list, colour_list, treatment_times, 
        treatment_labels=[], time_no_effect_mt=8*60
        ):
    # --- Plot probability with time -----
    times_to_plot = np.linspace(0, time_no_effect, 20)
    # figsize is fudged to make it approx. same height as other plot
    fig_probs_time, ax_probs_time = plt.subplots(figsize=(8,4))
    # probs_time_plot_title = (f'Treating {occlusion_str} with {treatment_str}' +
    #     f'\nat treatment time {time_input} minutes')
    plot_probs_filled(
        A_list, b_list, times_to_plot, colour_list, 
        # probs_to_mark=np.unique(probs_to_mark), 
        treatment_times, treatment_labels,
        ax=ax_probs_time, xmax=time_no_effect_mt/60)
    st.pyplot(fig_probs_time)


def build_latex_cumsum_string(
        weighted_added_utils, 
        mRS_list_time_input_treatment,
        mRS_list_no_treatment,
        mRS_dist_mix,
        colour_list, 
        utility_weights,
        util=True): 
    cumulative_changes = 0.0
    big_p_str = r'''\begin{align*}'''
    # Add column headings:
    big_p_str += (r'''& \mathrm{Treated} & & \mathrm{Not\ treated} & \mathrm{Proportion} \\
                    ''')
    for i in range(1, len(weighted_added_utils)):
        if weighted_added_utils[i]-weighted_added_utils[i-1]!=0:

            if util==True:
                value_treated = utility_weights[mRS_list_time_input_treatment[i]]
                value_no_treatment = utility_weights[mRS_list_no_treatment[i]]
                
                p_str_treated = f'{value_treated:5.2f}'
                p_str_no_treatment = f'{value_no_treatment:5.2f}'
            
            else:
                value_treated = mRS_list_time_input_treatment[i]
                value_no_treatment = mRS_list_no_treatment[i]

                p_str_treated = f'{value_treated:1d}'
                p_str_no_treatment = f'{value_no_treatment:1d}'
            
            bin_width = mRS_dist_mix[i]-mRS_dist_mix[i-1]

            p_str = ''

            # First weight: 
            p_str += r'''(&\textcolor{'''
            p_str += f'{colour_list[mRS_list_time_input_treatment[i]]}'
            p_str += r'''}{'''
            if value_treated>=0:
                # Add sneaky + for alignment
                p_str += r'\phantom{+}'
            p_str += p_str_treated
            p_str += r'''}&-&\textcolor{'''

            # Second weight: 
            p_str += f'{colour_list[mRS_list_no_treatment[i]]}'
            p_str += r'''}{'''
            if value_no_treatment >=0:
                # Add sneaky + for alignment
                p_str += r'\phantom{+}'
            p_str += p_str_no_treatment
            p_str += r'''}\phantom{+})& \times '''
            # Bin widths:
            p_str += f'{bin_width:5.3f}'
            # Value of this line:
            p_str += r''' = &'''
            value_here = (
                (value_treated - value_no_treatment) * bin_width
            )
            cumulative_changes += value_here

            if value_here>=0:
                # Add sneaky + for alignment
                p_str += r'\phantom{+}'
            p_str += f'{value_here:5.3f}'
            # Next line:
            p_str += '\\\\'

            # Add to the big string: 
            big_p_str += p_str 

    # Add total beneath the rest: 
    # big_p_str += r'''& & & & & & -----\\\\'''
    big_p_str += r'''\hline''' 
    big_p_str += r'''& & & & \mathrm{Total}: &'''

    if cumulative_changes>=0:
        # Add sneaky + for alignment
        big_p_str += r'\phantom{+}'
    big_p_str += f'{cumulative_changes:5.3f}\\\\'
    big_p_str += r'''\end{align*}'''

    return big_p_str 

def do_prob_bars( 
        dist_dict,
        mRS_dist_mix, weighted_added_utils,
        mRS_list_time_input_treatment, 
        mRS_list_no_treatment, time_input, key_str=''
        ):
    # ----- Plot probability distributions ----- 

    fig_bars_change, ax_bars = plt.subplots(figsize=(8,2))

    bar_height = 0.5
    y_list = [2, 1, 0]
    plot_bars(
        [dist_dict['dist_pre_stroke'], 
         dist_dict['dist_time_input_treatment'], 
         dist_dict['dist_no_treatment']
         ], 
        [dist_dict['dist_cumsum_pre_stroke'], 
         dist_dict['dist_cumsum_time_input_treatment'], 
         dist_dict['dist_cumsum_no_treatment']
         ], 
        ax_bars, time_input, y_list, bar_height)

    # ylim_util_change = (
    #     plot_utility_chart(ax_util_change, mRS_dist_mix, weighted_added_utils) )


    # if st.checkbox('Mark changes between treated and no treatment',
    #                 key='checkbox_mark_lines_'+key_str):
    #     top_of_bottom_bar = y_list[2]+bar_height*0.5
    #     bottom_of_top_bar = y_list[1]-bar_height*0.5
    #     for prob in mRS_dist_mix:
    #         ax_bars.vlines(prob, bottom_of_top_bar, top_of_bottom_bar,
    #             color='silver', linestyle='-', zorder=0)
    #         ax_util_change.axvline(prob, color='silver', zorder=0)


    # if st.checkbox('Add mRS colours to utility line',
    #                 key='checkbox_mRS_colours_'+key_str):
    #     draw_mRS_colours_on_utility_chart(
    #         mRS_dist_mix, weighted_added_utils, 
    #         mRS_list_time_input_treatment, mRS_list_no_treatment, 
    #         ylim_util_change, ax_util_change
    #         )
    #     write_text_about_utility_colours = 1
    # else:
    #     write_text_about_utility_colours = 0


    # Extend xlims slightly to not cut off bar border colour. 
    for ax in [ax_bars]:
        ax.set_xlim(-5e-3, 1.0+5e-3)
        ax.set_xlabel('Probability')
        ax.set_xticks(np.arange(0,1.01,0.2))
        ax.set_xticks(np.arange(0,1.01,0.05), minor=True)


    time_input_str = f'{time_input//60}hr {time_input%60}min'
    st.write('We can draw some of the data from the table in the "mRS distributions at the treatment times" section above to create these bar charts of mRS probability distributions:')

    st.pyplot(fig_bars_change)

    st.write('The weighted mean utility and mRS is calculated using those regions of the chart where the mRS is different between the "No treatment" and "Treated at '+time_input_str+'" bars.')

    # st.write('When the value of "probability" lands in different mRS bins between the two distributions, the graph has a slope. When the mRS values are the same, the graph is flat.')
    # if write_text_about_utility_colours>0:
    #     st.write('On either side of the cumulative weighted added utility (the solid black line), the two mRS distributions are shown. Immediately above the line is the treated distribution, and immediately below is the "no treatment" distribution.')


    st.write('Sums for the cumulative weighted utility:') 
    big_p_str = build_latex_cumsum_string(
            weighted_added_utils, 
            mRS_list_time_input_treatment,
            mRS_list_no_treatment,
            mRS_dist_mix,
            colour_list, 
            utility_weights,
            util=True)
    st.latex(big_p_str)

    # Check:
    # st.write(weighted_added_utils[-1])

    st.write('Sums for the cumulative weighted mRS:') 
    big_p_str = build_latex_cumsum_string(
            weighted_added_utils, 
            mRS_list_time_input_treatment,
            mRS_list_no_treatment,
            mRS_dist_mix,
            colour_list, 
            utility_weights,
            util=False)
    st.latex(big_p_str)



def compare_probs_with_time(dict1, dict2):
    # Plot probs with time 
    st.subheader('Probability variation with time')

    st.write('The mRS probability distributions are a concoction of various sources of data. The full details are given in [this document: "mRS distributions..."](https://github.com/samuel-book/stroke_outcome/blob/main/mRS_datasets_full.ipynb)')
    
    st.write('The boundaries between the mRS bins follow the shape of a logistic function. For details, see [this document: "Mathematics..."](https://github.com/samuel-book/stroke_outcome/blob/main/mRS_outcomes_maths.ipynb).')

    do_probs_with_time(
        dict1['time_no_effect'], dict1['A_list'], dict2['b_list'], 
        colour_list, 
        [dict1['treatment_time'], dict2['treatment_time']], 
        treatment_labels = [f'Case {i+1}' for i in range(2)],
        time_no_effect_mt=time_no_effect_mt
        )

    # # Tabulate mRS bins 
    st.subheader('mRS data tables')
    st.write('This table contains the probability distributions at key points from the probability vs. time graph above.')
    df_combo = make_combo_mRS_bin_dataframe(
        dict1['df_dists_bins'], dict2['df_dists_bins'], 
        dict1['treatment_time'], dict2['treatment_time'])
    st.table(df_combo)



# ###########################
# ##### START OF SCRIPT #####
# ###########################

# ----- Fixed parameters ----- 
# Define maximum treatment times:
time_no_effect_ivt = int(6.3*60)   # minutes
time_no_effect_mt = int(8*60)      # minutes


# Change default colour scheme:
plt.style.use('seaborn-colorblind')
colour_list = make_colour_list()

# Define some emoji for various situations:
emoji_dict = dict( 
    # onset = ':thumbsdown:',
    onset_to_ambulance_arrival = ':ambulance:',
    travel_to_ivt = ':hospital:', 
    travel_to_mt = ':hospital:', 
    ivt_arrival_to_treatment = ':pill:',
    transfer_additional_delay = ':ambulance:',#':hourglass_flowing_sand:',
    travel_ivt_to_mt = ':hospital:',
    mt_arrival_to_treatment = ':syringe:',
)


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
nlvo_ivt_case1_dict = find_useful_dist_dict(
    'Non-large vessel (nLVO)', 'Intravenous thrombolysis (IVT)', 
    case1_time_to_ivt
    )
nlvo_ivt_case2_dict = find_useful_dist_dict(
    'Non-large vessel (nLVO)', 'Intravenous thrombolysis (IVT)', 
    case2_time_to_ivt
    )
lvo_ivt_case1_dict = find_useful_dist_dict(
    'Large vessel (LVO)', 'Intravenous thrombolysis (IVT)',
    case1_time_to_ivt
    )
lvo_ivt_case2_dict = find_useful_dist_dict(
    'Large vessel (LVO)', 'Intravenous thrombolysis (IVT)', 
    case2_time_to_ivt
    )
lvo_mt_case1_dict = find_useful_dist_dict(
    'Large vessel (LVO)', 'Mechanical thrombectomy (MT)', 
    case1_time_to_mt
    )
lvo_mt_case2_dict = find_useful_dist_dict(
    'Large vessel (LVO)', 'Mechanical thrombectomy (MT)', 
    case2_time_to_mt
    )
# ich_case1_dict = find_useful_dist_dict(occlusion_input, treatment_input, time_input)
# ich_case2_dict = find_useful_dist_dict(occlusion_input, treatment_input, time_input)



mean_mRS_dict_nlvo_ivt_case1, mean_util_dict_nlvo_ivt_case1 = \
    calculate_mean_changes(
        nlvo_ivt_case1_dict['dist_pre_stroke'], 
        nlvo_ivt_case1_dict['dist_no_treatment'], 
        nlvo_ivt_case1_dict['dist_time_input_treatment'], 
        utility_weights
        )
mean_mRS_dict_nlvo_ivt_case2, mean_util_dict_nlvo_ivt_case2 = \
    calculate_mean_changes(
        nlvo_ivt_case2_dict['dist_pre_stroke'], 
        nlvo_ivt_case2_dict['dist_no_treatment'], 
        nlvo_ivt_case2_dict['dist_time_input_treatment'], 
        utility_weights
        )


mean_mRS_dict_lvo_ivt_case1, mean_util_dict_lvo_ivt_case1 = \
    calculate_mean_changes(
        lvo_ivt_case1_dict['dist_pre_stroke'], 
        lvo_ivt_case1_dict['dist_no_treatment'], 
        lvo_ivt_case1_dict['dist_time_input_treatment'], 
        utility_weights
        )
mean_mRS_dict_lvo_ivt_case2, mean_util_dict_lvo_ivt_case2 = \
    calculate_mean_changes(
        lvo_ivt_case2_dict['dist_pre_stroke'], 
        lvo_ivt_case2_dict['dist_no_treatment'], 
        lvo_ivt_case2_dict['dist_time_input_treatment'], 
        utility_weights
        )


mean_mRS_dict_lvo_mt_case1, mean_util_dict_lvo_mt_case1 = \
    calculate_mean_changes(
        lvo_mt_case1_dict['dist_pre_stroke'], 
        lvo_mt_case1_dict['dist_no_treatment'], 
        lvo_mt_case1_dict['dist_time_input_treatment'], 
        utility_weights
        )
mean_mRS_dict_lvo_mt_case2, mean_util_dict_lvo_mt_case2 = \
    calculate_mean_changes(
        lvo_mt_case2_dict['dist_pre_stroke'], 
        lvo_mt_case2_dict['dist_no_treatment'], 
        lvo_mt_case2_dict['dist_time_input_treatment'], 
        utility_weights
        )


mean_mRS_change_case1 = find_weighted_change(
    mean_mRS_dict_lvo_ivt_case1['diff_no_treatment'], 
    mean_mRS_dict_lvo_mt_case1['diff_no_treatment'], 
    mean_mRS_dict_nlvo_ivt_case1['diff_no_treatment'], 
    prop_dict, util=False)

mean_mRS_change_case2 = find_weighted_change(
    mean_mRS_dict_lvo_ivt_case2['diff_no_treatment'], 
    mean_mRS_dict_lvo_mt_case2['diff_no_treatment'], 
    mean_mRS_dict_nlvo_ivt_case2['diff_no_treatment'], 
    prop_dict, util=False)


mean_util_change_case1 = find_weighted_change(
    mean_util_dict_lvo_ivt_case1['diff_no_treatment'], 
    mean_util_dict_lvo_mt_case1['diff_no_treatment'], 
    mean_util_dict_nlvo_ivt_case1['diff_no_treatment'], 
    prop_dict)

mean_util_change_case2 = find_weighted_change(
    mean_util_dict_lvo_ivt_case2['diff_no_treatment'], 
    mean_util_dict_lvo_mt_case2['diff_no_treatment'], 
    mean_util_dict_nlvo_ivt_case2['diff_no_treatment'], 
    prop_dict)


# Find mean population utility and mRS with no treatment: 
mean_mRS_no_treatment_case1 = calculate_combo_mean_changes(
    prop_dict, 
    mean_mRS_dict_nlvo_ivt_case1, 
    mean_mRS_dict_lvo_ivt_case1, 
    mean_mRS_dict_lvo_mt_case1, 
    'no_treatment')
mean_mRS_no_treatment_case2 = calculate_combo_mean_changes(
    prop_dict, 
    mean_mRS_dict_nlvo_ivt_case2, 
    mean_mRS_dict_lvo_ivt_case2, 
    mean_mRS_dict_lvo_mt_case2, 
    'no_treatment')

mean_util_no_treatment_case1 = calculate_combo_mean_changes(
    prop_dict, 
    mean_util_dict_nlvo_ivt_case1, 
    mean_util_dict_lvo_ivt_case1, 
    mean_util_dict_lvo_mt_case1, 
    'no_treatment')
mean_util_no_treatment_case2 = calculate_combo_mean_changes(
    prop_dict, 
    mean_util_dict_nlvo_ivt_case2, 
    mean_util_dict_lvo_ivt_case2, 
    mean_util_dict_lvo_mt_case2, 
    'no_treatment')


mean_mRS_treated_case1 = mean_mRS_no_treatment_case1 + mean_mRS_change_case1
mean_mRS_treated_case2 = mean_mRS_no_treatment_case2 + mean_mRS_change_case2

mean_util_treated_case1 = mean_util_no_treatment_case1 + mean_util_change_case1
mean_util_treated_case2 = mean_util_no_treatment_case2 + mean_util_change_case2


# ###########################
# ######### RESULTS #########
# ###########################
st.header('Results')
# ----- Show metric for +/- mRS and utility -----
st.subheader('Changes in mRS and utility')

# Put the two metrics in columns: 
met_col1, met_col2 = st.columns(2)

# mRS:
met_col1.subheader('--- Case 1 ---')

met_col1.write(
    f'💊 Time to IVT: {case1_time_to_ivt//60}hr {case1_time_to_ivt%60}min.'
    )
met_col1.write(
    f'💉 Time to MT: {case1_time_to_mt//60}hr {case1_time_to_mt%60}min.'
    )
met_col1.write('-'*20)

met_col1.metric('Population mean mRS', 
    f'{mean_mRS_treated_case1:0.2f}', 
    f'{mean_mRS_change_case1:0.2f} from no treatment',
    delta_color='inverse' # A negative difference is good.
    )
met_col1.write('-'*20)
met_col1.metric('Population mean utility', 
    f'{mean_util_treated_case1:0.3f}', 
    f'{mean_util_change_case1:0.3f} from no treatment',
    )

# Utility: 
met_col2.subheader('--- Case 2 ---')

met_col2.write(
    f'💊 Time to IVT: {case2_time_to_ivt//60}hr {case2_time_to_ivt%60}min.'
    )
met_col2.write(
    f'💉 Time to MT: {case2_time_to_mt//60}hr {case2_time_to_mt%60}min.'
    )
met_col2.write('-'*20)

met_col2.metric('Population mean mRS', 
    f'{mean_mRS_treated_case2:0.2f}', 
    f'{mean_mRS_change_case2:0.2f} from no treatment',
    delta_color='inverse' # A negative difference is good.
    )
met_col2.write('-'*20)
met_col2.metric('Population mean utility', 
    f'{mean_util_treated_case2:0.3f}', 
    f'{mean_util_change_case2:0.3f} from no treatment',
    )


# ###########################
# ######### METHOD ##########
# ###########################
st.write('-'*50)

st.header('Details of the calculation')
st.write('The following bits detail the calculation.')




# ----- Probability distributions ----- 
st.subheader('mRS distributions at the treatment times')
probdist_expander = st.expander('Show probability distributions')
with probdist_expander: 

    # ----- Add legend -----
    fig_legend = make_fig_legend(colour_list)
    st.pyplot(fig_legend)

    tab1, tab2, tab3, tab4 = st.tabs([
        'nLVO treated with IVT',
        'LVO treated with IVT only',
        'LVO treated with MT',
        'ICH' 
    ])

    with tab1:
        compare_probs_with_time(nlvo_ivt_case1_dict, nlvo_ivt_case2_dict)
    with tab2:
        compare_probs_with_time(lvo_ivt_case1_dict, lvo_ivt_case2_dict)
    with tab3:
        compare_probs_with_time(lvo_mt_case1_dict, lvo_mt_case2_dict)
    with tab4:
        st.write('Nothing to see here.')




# ----- Probability distributions ----- 
st.subheader('Cumulative changes in mRS and utility')
cumulative_expander = st.expander('Show cumulative changes')
with cumulative_expander: 

    # ----- Add legend -----
    fig_legend = make_fig_legend(colour_list)
    st.pyplot(fig_legend)

    tab1, tab2, tab3, tab4 = st.tabs([
        'nLVO treated with IVT',
        'LVO treated with IVT only',
        'LVO treated with MT',
        'ICH' 
    ])

    with tab1:
        # nLVO IVT 

        st.subheader('The effect of treatment on mRS')
        st.subheader('Case 1')
        # Case 1: 
        mRS_dist_mix, weighted_added_utils, mRS_list_time_input_treatment, mRS_list_no_treatment = \
            find_added_utility_between_dists(
                nlvo_ivt_case1_dict['dist_cumsum_time_input_treatment'], 
                nlvo_ivt_case1_dict['dist_cumsum_no_treatment']
                )
        do_prob_bars( 
            nlvo_ivt_case1_dict,
            mRS_dist_mix, 
            weighted_added_utils,
            mRS_list_time_input_treatment, 
            mRS_list_no_treatment,
            case1_time_to_ivt,
            key_str = 'nLVO_IVT_case1')

        st.subheader('Case 2')
        # Case 2: 
        mRS_dist_mix, weighted_added_utils, mRS_list_time_input_treatment, mRS_list_no_treatment = \
            find_added_utility_between_dists(
                nlvo_ivt_case2_dict['dist_cumsum_time_input_treatment'], 
                nlvo_ivt_case2_dict['dist_cumsum_no_treatment']
                )
        do_prob_bars( 
            nlvo_ivt_case2_dict,
            mRS_dist_mix, 
            weighted_added_utils,
            mRS_list_time_input_treatment, 
            mRS_list_no_treatment,
            case2_time_to_ivt,
            key_str = 'nLVO_IVT_case2')


        #compare_probs_with_time(nlvo_ivt_case1_dict, nlvo_ivt_case2_dict)
    with tab2:
        # LVO IVT 
        st.subheader('The effect of treatment on mRS')
        st.subheader('Case 1')
        # Case 1:
        mRS_dist_mix, weighted_added_utils, mRS_list_time_input_treatment, mRS_list_no_treatment = \
            find_added_utility_between_dists(
                lvo_ivt_case1_dict['dist_cumsum_time_input_treatment'], 
                lvo_ivt_case1_dict['dist_cumsum_no_treatment']
                )
        do_prob_bars( 
            lvo_ivt_case1_dict,
            mRS_dist_mix, 
            weighted_added_utils,
            mRS_list_time_input_treatment, 
            mRS_list_no_treatment,
            case1_time_to_ivt,
            key_str = 'LVO_IVT_case1') 


        st.subheader('Case 2')
        # Case 2: 
        mRS_dist_mix, weighted_added_utils, mRS_list_time_input_treatment, mRS_list_no_treatment = \
            find_added_utility_between_dists(
                lvo_ivt_case2_dict['dist_cumsum_time_input_treatment'], 
                lvo_ivt_case2_dict['dist_cumsum_no_treatment']
                )
        do_prob_bars( 
            lvo_ivt_case2_dict,
            mRS_dist_mix, 
            weighted_added_utils,
            mRS_list_time_input_treatment, 
            mRS_list_no_treatment,
            case2_time_to_ivt,
            key_str = 'LVO_IVT_case2')

        #compare_probs_with_time(lvo_ivt_case1_dict, lvo_ivt_case2_dict)
    with tab3:
        # LVO MT 
        st.subheader('The effect of treatment on mRS')
        st.subheader('Case 1')
        # Case 1:
        mRS_dist_mix, weighted_added_utils, mRS_list_time_input_treatment, mRS_list_no_treatment = \
            find_added_utility_between_dists(
                lvo_mt_case1_dict['dist_cumsum_time_input_treatment'], 
                lvo_mt_case1_dict['dist_cumsum_no_treatment']
                )
        do_prob_bars( 
            lvo_mt_case1_dict,
            mRS_dist_mix, 
            weighted_added_utils,
            mRS_list_time_input_treatment, 
            mRS_list_no_treatment,
            case1_time_to_mt,
            key_str = 'LVO_MT_case1') 


        st.subheader('Case 2')
        # Case 2: 
        mRS_dist_mix, weighted_added_utils, mRS_list_time_input_treatment, mRS_list_no_treatment = \
            find_added_utility_between_dists(
                lvo_mt_case2_dict['dist_cumsum_time_input_treatment'], 
                lvo_mt_case2_dict['dist_cumsum_no_treatment']
                )
        do_prob_bars( 
            lvo_mt_case2_dict,
            mRS_dist_mix, 
            weighted_added_utils,
            mRS_list_time_input_treatment, 
            mRS_list_no_treatment,
            case2_time_to_mt,
            key_str = 'LVO_MT_case2')
        #compare_probs_with_time(lvo_mt_case1_dict, lvo_mt_case2_dict)
    with tab4:
        st.write('Nothing to see here.')





# ----- Sum up changes ----- 
st.subheader('Calculations for overall changes in utility and mRS')
total_expander = st.expander('Show the sums')
with total_expander: 
    st.write('For each group, the weighted change is equal to the product '+
             'of the following:')
    st.write('+ proportion with this stroke type (%)')
    st.write('+ proportion receiving this treatment (%)')
    st.write('+ total change across this population')
    st.write('The final change given in the Results section above ' +
             'is the sum of the weighted changes.')

    do_change_sums(
        prop_dict, 
        mean_mRS_dict_nlvo_ivt_case1,
        mean_mRS_dict_lvo_ivt_case1,
        mean_mRS_dict_lvo_mt_case1,
        mean_mRS_change_case1,
        mean_mRS_dict_nlvo_ivt_case2,
        mean_mRS_dict_lvo_ivt_case2,
        mean_mRS_dict_lvo_mt_case2,
        mean_mRS_change_case2,
        mean_util_dict_nlvo_ivt_case1,
        mean_util_dict_lvo_ivt_case1,
        mean_util_dict_lvo_mt_case1,
        mean_util_change_case1,
        mean_util_dict_nlvo_ivt_case2,
        mean_util_dict_lvo_ivt_case2,
        mean_util_dict_lvo_mt_case2,
        mean_util_change_case2,
        )



    # st.latex(r'''\textcolor{#0072B2}{Hello}''')
    # st.latex(r'''\textcolor{#009E73}{Hello}''')
    # st.latex(r'''\textcolor{#D55E00}{Hello}''')
    # st.latex(r'''\textcolor{#CC79A7}{Hello}''')
    # st.latex(r'''\textcolor{#F0E442}{Hello}''')
    # st.latex(r'''\textcolor{#56B4E9}{Hello}''')
    # st.latex(r'''\textcolor{DarkSlateGray}{Hello}''')
