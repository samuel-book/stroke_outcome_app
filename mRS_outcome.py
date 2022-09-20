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
from outcome_utilities.dist_plot import \
    draw_horizontal_bar, draw_connections
from outcome_utilities.bar_sizes import \
    find_bin_size_ratios
from outcome_utilities.added_utility_between_dists import \
    find_added_utility_between_dists


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

def find_mRS_dists_from_file(occlusion_str, treatment_str):
    """
    Import the mRS data from file.
    
    If no data exists, return warning string.
    """
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

    try:
        # Get the dists from the data array using the strings: 
        dist_pre_stroke = mrs_dists_bins.loc[dist_pre_stroke_str].values
        dist_no_treatment = mrs_dists_bins.loc[dist_no_treatment_str].values
        dist_t0_treatment = mrs_dists_bins.loc[dist_t0_treatment_str].values
        dist_no_effect = mrs_dists_bins.loc[dist_no_effect_str].values

        dist_cumsum_pre_stroke = mrs_dists_cumsum.loc[dist_pre_stroke_str].values
        dist_cumsum_no_treatment = (
            mrs_dists_cumsum.loc[dist_no_treatment_str].values)
        dist_cumsum_t0_treatment = (
            mrs_dists_cumsum.loc[dist_t0_treatment_str].values)
        dist_cumsum_no_effect = mrs_dists_cumsum.loc[dist_no_effect_str].values
        return [dist_pre_stroke, dist_no_treatment, 
            dist_t0_treatment, dist_no_effect,
            dist_cumsum_pre_stroke, dist_cumsum_no_treatment, 
            dist_cumsum_t0_treatment, dist_cumsum_no_effect]
    except:
        err_str = (f':warning: No data for {occlusion_str} and '+
            f'{treatment_str.upper()}.')
        return err_str


def find_dists_at_chosen_time(dist_cumsum_t0_treatment, 
    dist_cumsum_no_effect, time_input, time_no_effect):
    """
    Find the mRS distributions at the input treatment time.
    Finds the cumulative and non-cumulative probability distributions.

    Return the A and b lists for later plotting. 
    """
    a_list, b_list, A_list = find_mrs_constants(
        dist_cumsum_t0_treatment, dist_cumsum_no_effect, time_no_effect)
    # The probability P at an arbitrary treatment time is: 
    # P(treatment_time) = 1.0/(1.0 + np.exp(-A -b*treatment_time)) 
    # where the values of a, b, and A are different for each mRS. 


    # ----- Find distribution at chosen treatment time -----
    # Calculate the cumulative probability bins at the chosen time: 
    dist_cumsum_time_input_treatment = (
        find_mrs_bins_t(A_list, b_list, time_input))
    # Append the value for mRS=6:
    dist_cumsum_time_input_treatment = (
        np.append(dist_cumsum_time_input_treatment, 1.0))
    # Make the non-cumulative bins: 
    dist_time_input_treatment = (
        np.diff(dist_cumsum_time_input_treatment, prepend=0.0))
    return (dist_time_input_treatment, dist_cumsum_time_input_treatment,
        A_list, b_list)


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

# ###########################
# ##### START OF SCRIPT #####
# ###########################

# ----- Fixed parameters ----- 
# Define maximum treatment times:
time_no_effect_ivt = int(6.3*60)   # minutes
time_no_effect_mt = int(8*60)      # minutes

# Utility for each mRS:
utility_weights = np.array([0.97, 0.88, 0.74, 0.55, 0.20, -0.19, 0.00])

# Change default colour scheme:
plt.style.use('seaborn-colorblind')
colour_list = make_colour_list()


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

# Define the time of no effect *after* the treatment input. 
time_no_effect = (
    time_no_effect_mt if 'MT' in treatment_input else time_no_effect_ivt)

time_input = st.sidebar.slider(
    'Choose the treatment time in minutes.',
    0, time_no_effect, 60 # start, end, default value
    )
# Convert to hours and minutes: 
treatment_time_str = (
    f'Treatment time: {time_input//60} hours {time_input%60} minutes')
st.sidebar.write(treatment_time_str)

# Use the inputs to set up some strings for importing data:
occlusion_str = 'nlvo' if 'nLVO' in occlusion_input else 'lvo'
treatment_str = 'mt' if 'MT' in treatment_input else 'ivt'
excess_death_str = '_'+treatment_str+'_deaths' 



# ----- Select the mRS data -----
all_dists = find_mRS_dists_from_file(occlusion_str, treatment_str)
if type(all_dists)==str:
    # No data was imported, so print a warning message:
    st.warning(all_dists)
    st.stop()
else:
    dist_pre_stroke, dist_no_treatment, \
    dist_t0_treatment, dist_no_effect, \
    dist_cumsum_pre_stroke, dist_cumsum_no_treatment, \
    dist_cumsum_t0_treatment, dist_cumsum_no_effect = all_dists


# Use the inputs to print a new header: 
title = (f'Treating {occlusion_str} with {treatment_str.upper()} ' +
    f'at treatment time {time_input//60} hours {time_input%60} minutes')
st.header(title)


# ----- Find probability with time -----
(dist_time_input_treatment, dist_cumsum_time_input_treatment, 
    A_list, b_list) = find_dists_at_chosen_time(dist_cumsum_t0_treatment, 
        dist_cumsum_no_effect, time_input, time_no_effect)


#  ----- Make data frames -----
# Set up headings for the rows and columns: 
headings_rows = [
    'Pre-stroke',
    'Treatment at 0 hours',
    f'Treatment at {time_input//60} hours {time_input%60} minutes',
    f'Treatment at {time_no_effect//60} hours {time_no_effect%60} minutes',
    'Untreated'
    ]
headings_cols_cumsum = [f'mRS<={i}' for i in range(7)]
headings_cols_bins = [f'mRS={i}' for i in range(7)]

# Build data frames: 
df_dists_cumsum = pd.DataFrame(
    [
    dist_cumsum_pre_stroke, 
    dist_cumsum_t0_treatment, 
    dist_cumsum_time_input_treatment,
    dist_cumsum_no_effect, 
    dist_cumsum_no_treatment
    ], 
    index=headings_rows, columns=headings_cols_cumsum)
df_dists_bins = pd.DataFrame(
    [
    dist_pre_stroke, 
    dist_t0_treatment, 
    dist_time_input_treatment,
    dist_no_effect, 
    dist_no_treatment
    ],
    index=headings_rows, columns=headings_cols_bins)


# ----- Find cumulative added utility -----
(mRS_dist_mix, weighted_added_utils, mRS_list_time_input_treatment, 
    mRS_list_no_treatment) = find_added_utility_between_dists(
    dist_cumsum_time_input_treatment, dist_cumsum_no_treatment,
    utility_weights
)


# ----- Calculate metrics -----
# Calculate mean mRSes:
mean_mRS_pre_stroke = np.sum(
    dist_pre_stroke*np.arange(7))
mean_mRS_no_treatment = np.sum(dist_no_treatment*np.arange(7))
mean_mRS_time_input_treatment = np.sum(
    dist_time_input_treatment*np.arange(7))
# Differences:
mean_mRS_diff_no_treatment = (
    mean_mRS_time_input_treatment - mean_mRS_no_treatment)
mean_mRS_diff_pre_stroke = (
    mean_mRS_time_input_treatment - mean_mRS_pre_stroke)

# Calculate mean utilities: 
# (it seems weird to use "sum" instead of "mean" but this definition 
# matches the clinical outcome script)
mean_utility_pre_stroke = np.sum(
    dist_pre_stroke*utility_weights)
mean_utility_no_treatment = np.sum(dist_no_treatment*utility_weights)
mean_utility_time_input_treatment = np.sum(
    dist_time_input_treatment*utility_weights)
# Differences:
mean_utility_diff_no_treatment = (
    mean_utility_time_input_treatment - mean_utility_no_treatment)
mean_utility_diff_pre_stroke = (
    mean_utility_time_input_treatment - mean_utility_pre_stroke)


# ----- Add legend -----
# Make dummy columns to force legend to be smaller:
leg_cols = st.columns(3)
fig_legend = make_fig_legend(colour_list)
leg_cols[1].pyplot(fig_legend)


# -----
# Put the two big plots in columns: 
plot_col1, plot_col2 = st.columns(2)

# --- Plot probability with time -----
plot_col1.subheader('Probability changes with time')
times_to_plot = np.linspace(0, time_no_effect, 20)
# figsize is fudged to make it approx. same height as other plot
fig_probs_time, ax_probs_time = plt.subplots(figsize=(8,7.3))
# probs_time_plot_title = (f'Treating {occlusion_str} with {treatment_str}' +
#     f'\nat treatment time {time_input} minutes')
plot_probs_filled(A_list, b_list, times_to_plot, colour_list, 
    # probs_to_mark=np.unique(probs_to_mark), 
    time_input,
    ax=ax_probs_time)
plot_col1.pyplot(fig_probs_time)


# ----- Plot probability distributions ----- 
plot_col2.subheader('The effect of treatment on mRS')

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


if plot_col2.checkbox('Mark changes between treated and no treatment'):
    top_of_bottom_bar = y_list[2]+bar_height*0.5
    bottom_of_top_bar = y_list[1]-bar_height*0.5
    for prob in mRS_dist_mix:
        ax_bars.vlines(prob, bottom_of_top_bar, top_of_bottom_bar,
            color='silver', linestyle='-', zorder=0)
        ax_util_change.axvline(prob, color='silver', zorder=0)


if plot_col2.checkbox('Add mRS colours to utility line'):
    draw_mRS_colours_on_utility_chart(mRS_dist_mix, weighted_added_utils, 
        mRS_list_time_input_treatment, mRS_list_no_treatment, 
        ylim_util_change)
    write_text_about_utility_colours = 1
else:
    write_text_about_utility_colours = 0


# Extend xlims slightly to not cut off bar border colour. 
for ax in axs:
    ax.set_xlim(-5e-3, 1.0+5e-3)
    ax.set_xlabel('Probability')
    ax.set_xticks(np.arange(0,1.01,0.2))
    ax.set_xticks(np.arange(0,1.01,0.05), minor=True)


plot_col2.pyplot(fig_bars_change)

plot_col2.write('When the value of "probability" lands in different mRS bins between the two distributions, the graph has a slope. When the mRS values are the same, the graph is flat.')
if write_text_about_utility_colours>0:
    plot_col2.write('On either side of the cumulative weighted added utility (the solid black line), the two mRS distributions are shown. Immediately above the line is the treated distribution, and immediately below is the "no treatment" distribution.')



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


# ----- Show data frame -----
st.subheader('mRS data tables')
# If box is ticked, display the tables: 
dataframe_expander = st.expander('Show data tables')
with dataframe_expander:
    st.write('Probability bins:')
    st.dataframe(df_dists_bins)

    st.write('Cumulative probability bins:')
    st.dataframe(df_dists_cumsum)