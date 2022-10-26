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
    find_added_utility_between_dists, calculate_mean_changes
from outcome_utilities.inputs import \
    find_useful_dist_dict, inputs_patient_population, inputs_pathway


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


def plot_utility_chart(ax_util_change, mRS_dist_mix, weighted_added_utils,
    occlusion_str, treatment_str):
    # Bottom subplot
    if occlusion_str=='nlvo':
        ylim_util_change = [-0.02, 0.162] # nLVO IVT
    else:
        if treatment_str=='mt':
            ylim_util_change = [-0.05, 0.310] # LVO MT
        else:
            ylim_util_change = [-0.027, 0.062] # LVO IVT

    ax_util_change.plot(mRS_dist_mix, weighted_added_utils, color='k', 
        label='Cumulative weighted added utility')

    ax_util_change.set_ylabel('Cumulative\nweighted added utility')
    ax_util_change.set_xlabel('Probability')
    ax_util_change.tick_params(top=True, right=True, which='both')

    ax_util_change.set_ylim(ylim_util_change)
    return ylim_util_change
    
    
def plot_bars(dists_to_bar, dists_cumsum_to_bar, ax_bars, treatment_str, 
    time_input, y_list, bar_height):
    y_labels = [
        'Pre-stroke', 
        (f'Treated with {treatment_str.upper()} at '+'\n'
        + f'{time_input//60} hours '
        + f'{time_input%60:02d} minutes'),
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
    mRS_list_time_input_treatment, mRS_list_no_treatment, ylim_util_change,):
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


def plot_timeline(time_dict, ax=None, y=0):
        if ax==None:
            fig, ax = plt.subplots()
        time_cumulative = 0.0
        y_emoji_offset = 0.05
        y_under_offset = -0.05
        for time_key in time_dict.keys():
            t_min = time_dict[time_key]
            time_cumulative += t_min/60.0

            if 'ivt_arrival_to_treatment' in time_key:
                colour = 'b' 
                write_under=True 
            elif 'mt_arrival_to_treatment' in time_key:
                colour = 'r'
                write_under=True 
            else:
                colour = 'k'
                write_under=False 

            if time_dict[time_key]==0.0 and time_key!='onset':
                x_plot = np.NaN 
            else:
                x_plot = time_cumulative
            ax.scatter(x_plot, y, marker='|', s=200, color=colour)

            ax.annotate(
                time_key, xy=(x_plot, y+y_emoji_offset), 
                rotation=50, color=colour)
            if write_under: 
                time_str = (f'{int(60*time_cumulative//60):2d}hr '+
                            f'{int(60*time_cumulative%60):2d}min')
                ax.annotate(
                    time_str, xy=(x_plot, y+y_under_offset), color=colour,
                    ha='center', va='top')
            # ax.annotate(emoji_dict[time_key], xy=(time_cumulative, y+y_emoji_offset))
        ax.plot([0, time_cumulative], [y,y], color='k', zorder=0)

def make_timeline_plot(ax, time_dicts):
    
    y_step = 0.5
    y_vals = np.arange(0.0, y_step*len(time_dicts), y_step)[::-1]
    for i, time_dict in enumerate(time_dicts):
        plot_timeline(time_dict, ax, y=y_vals[i])

    xlim = ax.get_xlim()
    ax.set_xticks(np.arange(0, xlim[1], (10/60)), minor=True)
    ax.set_xlabel('Time since onset (hours)')

    ax.set_ylim(-0.1, y_step*len(time_dicts))
    ax.set_yticks(y_vals)
    ax.set_yticklabels(
        [f'Case {i+1}' for i in range(len(time_dicts))], fontsize=14)
    for spine in ['top', 'left', 'right']:
        ax.spines[spine].set_color('w')





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

def do_prob_bars( 
        dist_pre_stroke, dist_time_input_treatment, 
        dist_no_treatment, dist_cumsum_pre_stroke, 
        dist_cumsum_time_input_treatment, dist_cumsum_no_treatment,
        mRS_dist_mix, weighted_added_utils,
        treatment_str, occlusion_str, mRS_list_time_input_treatment, 
        mRS_list_no_treatment
        ):
    # ----- Plot probability distributions ----- 
    st.subheader('The effect of treatment on mRS')

    fig_bars_change, axs = plt.subplots(2, 1, figsize=(8,8))
    ax_bars = axs[0] 
    ax_util_change = axs[1]


    bar_height = 0.5
    y_list = [2, 1, 0]
    plot_bars(
        [dist_pre_stroke, dist_time_input_treatment, dist_no_treatment], 
        [dist_cumsum_pre_stroke, dist_cumsum_time_input_treatment, 
            dist_cumsum_no_treatment], ax_bars, treatment_str, 
        time_input, y_list, bar_height)
    ylim_util_change = (
        plot_utility_chart(ax_util_change, mRS_dist_mix, weighted_added_utils,
        occlusion_str, treatment_str) )


    if st.checkbox('Mark changes between treated and no treatment'):
        top_of_bottom_bar = y_list[2]+bar_height*0.5
        bottom_of_top_bar = y_list[1]-bar_height*0.5
        for prob in mRS_dist_mix:
            ax_bars.vlines(prob, bottom_of_top_bar, top_of_bottom_bar,
                color='silver', linestyle='-', zorder=0)
            ax_util_change.axvline(prob, color='silver', zorder=0)


    if st.checkbox('Add mRS colours to utility line'):
        draw_mRS_colours_on_utility_chart(
            mRS_dist_mix, weighted_added_utils, 
            mRS_list_time_input_treatment, mRS_list_no_treatment, 
            ylim_util_change
            )
        write_text_about_utility_colours = 1
    else:
        write_text_about_utility_colours = 0


    # Extend xlims slightly to not cut off bar border colour. 
    for ax in axs:
        ax.set_xlim(-5e-3, 1.0+5e-3)
        ax.set_xlabel('Probability')
        ax.set_xticks(np.arange(0,1.01,0.2))
        ax.set_xticks(np.arange(0,1.01,0.05), minor=True)


    st.pyplot(fig_bars_change)

    st.write('When the value of "probability" lands in different mRS bins between the two distributions, the graph has a slope. When the mRS values are the same, the graph is flat.')
    if write_text_about_utility_colours>0:
        st.write('On either side of the cumulative weighted added utility (the solid black line), the two mRS distributions are shown. Immediately above the line is the treated distribution, and immediately below is the "no treatment" distribution.')






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
    onset = ':thumbsdown:',
    onset_to_ambulance_arrival = ':ambulance:',
    travel_to_ivt = ':hospital:', 
    travel_to_mt = ':hospital:', 
    ivt_arrival_to_treatment = ':pill:',
    transfer_additional_delay = ':hourglass:',
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
# st.subheader('test')
st.write(
    'Emoji! :ambulance: :hospital: :pill: :syringe: ',
    ':hourglass_flowing_sand: :crystal_ball: :ghost: :skull: ',
    ':thumbsup: :thumbsdown:' 
    )


# ----- Population parameters -----
st.header('Patient population')
prop_dict = inputs_patient_population()


# ----- Timeline of patient pathway -----
st.header('Patient pathway')
st.write('Give all times in minutes. :ambulance:')
(case1_time_dict, case2_time_dict, case1_time_to_ivt, 
 case1_time_to_mt, case2_time_to_ivt, case2_time_to_mt) = inputs_pathway()

# Define the two cases: 
st.write('We can compare the expected outcomes for two cases. ')

st.write('__Case 1:__ all eligible patients receive IVT at the IVT-only centre, and then patients requiring MT are transported to the IVT+MT centre for further treatment.')
# # Summarise treatment times: 
# st.write(f'+ Time from onset to IVT is {case1_time_to_ivt//60} hours ',
#          f'{case1_time_to_ivt%60} minutes.')
# st.write(f'+ Time from onset to MT is {case1_time_to_mt//60} hours ',
#          f'{case1_time_to_mt%60} minutes.')

st.write('__Case 2:__ all patients are transported directly to the IVT+MT centre and receive the appropriate treatments there.')

# st.write(f'+ Time from onset to IVT is {case2_time_to_ivt//60} hours ',
#          f'{case2_time_to_ivt%60} minutes.')
# st.write(f'+ Time from onset to MT is {case2_time_to_mt//60} hours ',
#          f'{case2_time_to_mt%60} minutes.')

# Draw timelines 
fig, ax = plt.subplots()
make_timeline_plot(ax, [case1_time_dict, case2_time_dict])
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


def make_combo_mRS_bin_dataframe(df1, df2, treatment_time1, treatment_time2):
    if treatment_time1<treatment_time2:
        df_main = df1 
        df_extra = df2 
    elif treatment_time2<treatment_time1:
        df_main = df2 
        df_extra = df1 
    else:
        # Same rows in both so just return one: 
        return df1 

    new_df = pd.concat((df_main.iloc[:3], df_extra.iloc[2:3], df_main.iloc[3:]), axis=0)
    return new_df 

def compare_cases(dict1, dict2):
    # Plot probs with time 
    st.subheader('Probability variation with time')
    do_probs_with_time(
        dict1['time_no_effect'], dict1['A_list'], dict2['b_list'], 
        colour_list, 
        [dict1['treatment_time'], dict2['treatment_time']], 
        treatment_labels = [f'Case {i+1}' for i in range(2)],
        time_no_effect_mt=time_no_effect_mt
        )

    # # Plot change in util, mRS 
    # st.subheader('Changes in utility and mRS')
    # do_prob_bars()


    # # Tabulate mRS bins 
    st.subheader('mRS data tables')
    df_combo = make_combo_mRS_bin_dataframe(
        dict1['df_dists_bins'], dict2['df_dists_bins'], 
        dict1['treatment_time'], dict2['treatment_time'])
    st.dataframe(df_combo)

# ----- Probability distributions ----- 
st.header('Calculate changes to probability distributions')
probdist_expander = st.expander('Probability distributions')
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
        compare_cases(nlvo_ivt_case1_dict, nlvo_ivt_case2_dict)
    with tab2:
        compare_cases(lvo_ivt_case1_dict, lvo_ivt_case2_dict)
    with tab3:
        compare_cases(lvo_mt_case1_dict, lvo_mt_case2_dict)
    with tab4:
        st.write('Nothing to see here.')



# ----- Sum up changes ----- 
st.header('Calculate overall changes in utility and mRS')
total_expander = st.expander('Change sums')
with total_expander: 
    cols_util_tables = st.columns(2) 
    cols_util_tables[0].write('y')
    cols_util_tables[1].write('s')

    cols_mRS_tables = st.columns(2) 
    cols_mRS_tables[0].write('f')
    cols_mRS_tables[1].write('d')






mean_mRS_dict_nlvo_ivt_case1, mean_util_dict_nlvo_ivt_case1 = \
    calculate_mean_changes(
        nlvo_ivt_case1_dict['dist_pre_stroke'], 
        nlvo_ivt_case1_dict['dist_no_treatment'], 
        nlvo_ivt_case1_dict['dist_time_input_treatment'], 
        nlvo_ivt_case1_dict['utility_weights']
        )
mean_mRS_dict_nlvo_ivt_case2, mean_util_dict_nlvo_ivt_case2 = \
    calculate_mean_changes(
        nlvo_ivt_case2_dict['dist_pre_stroke'], 
        nlvo_ivt_case2_dict['dist_no_treatment'], 
        nlvo_ivt_case2_dict['dist_time_input_treatment'], 
        nlvo_ivt_case2_dict['utility_weights']
        )


mean_mRS_dict_lvo_ivt_case1, mean_util_dict_lvo_ivt_case1 = \
    calculate_mean_changes(
        lvo_ivt_case1_dict['dist_pre_stroke'], 
        lvo_ivt_case1_dict['dist_no_treatment'], 
        lvo_ivt_case1_dict['dist_time_input_treatment'], 
        lvo_ivt_case1_dict['utility_weights']
        )
mean_mRS_dict_lvo_ivt_case2, mean_util_dict_lvo_ivt_case2 = \
    calculate_mean_changes(
        lvo_ivt_case2_dict['dist_pre_stroke'], 
        lvo_ivt_case2_dict['dist_no_treatment'], 
        lvo_ivt_case2_dict['dist_time_input_treatment'], 
        lvo_ivt_case2_dict['utility_weights']
        )


mean_mRS_dict_lvo_mt_case1, mean_util_dict_lvo_mt_case1 = \
    calculate_mean_changes(
        lvo_mt_case1_dict['dist_pre_stroke'], 
        lvo_mt_case1_dict['dist_no_treatment'], 
        lvo_mt_case1_dict['dist_time_input_treatment'], 
        lvo_mt_case1_dict['utility_weights']
        )
mean_mRS_dict_lvo_mt_case2, mean_util_dict_lvo_mt_case2 = \
    calculate_mean_changes(
        lvo_mt_case2_dict['dist_pre_stroke'], 
        lvo_mt_case2_dict['dist_no_treatment'], 
        lvo_mt_case2_dict['dist_time_input_treatment'], 
        lvo_mt_case2_dict['utility_weights']
        )


mean_mRS_case1 = (
    prop_dict['prop_lvo']*prop_dict['lvo_treated_ivt_only']*mean_mRS_dict_lvo_ivt_case1['time_input_treatment'] +
    prop_dict['prop_lvo']*prop_dict['lvo_treated_ivt_only']*mean_mRS_dict_lvo_mt_case1['time_input_treatment'] +
    prop_dict['prop_nlvo']*prop_dict['nlvo_treated_ivt_only']*mean_mRS_dict_nlvo_ivt_case1['time_input_treatment'] 
)

    prop_dict = dict(
        nlvo = prop_nlvo,
        lvo = prop_lvo,
        ich = prop_ich,
        nlvo_treated_ivt_only = prop_nlvo_treated_ivt_only,
        lvo_treated_ivt_only = prop_lvo_treated_ivt_only,
        lvo_treated_ivt_mt = prop_lvo_treated_ivt_mt,
        ich_treated = prop_ich_treated,
        lvo_mt_also_receiving_ivt = prop_lvo_mt_also_receiving_ivt,
    )

    mean_util_dict = dict(
        pre_stroke = mean_utility_pre_stroke,
        no_treatment = mean_utility_no_treatment,
        time_input_treatment = mean_utility_time_input_treatment,
        diff_no_treatment = mean_utility_diff_no_treatment,
        diff_pre_stroke = mean_utility_diff_pre_stroke
    )

# ----- Show metric for +/- mRS and utility -----
st.subheader('Change in mRS and utility')

# Put the two metrics in columns: 
met_col1, met_col2 = st.columns(2)

# mRS:
met_col1.metric('Population mean mRS', 
    f'{mean_mRS_time_input_treatment:0.2f}', 
    f'{mean_mRS_diff_no_treatment:0.2f} from "no treatment"',
    delta_color='inverse' # A negative difference is good.
    )
met_col1.metric('',#'Population mean mRS', 
    '',#f'{mean_mRS_time_input_treatment:0.2f}', 
    f'{mean_mRS_diff_pre_stroke:0.2f} from "pre-stroke"',
    delta_color='inverse' # A negative difference is good.
    )

# Utility: 
met_col2.metric('Population mean utility', 
    f'{mean_utility_time_input_treatment:0.3f}', 
    f'{mean_utility_diff_no_treatment:0.3f} from "no treatment"',
    )
met_col2.metric('',#'Population mean utility', 
    '',#f'{mean_utility_time_input_treatment:0.3f}', 
    f'{mean_utility_diff_pre_stroke:0.3f} from "pre-stroke"',
    )

