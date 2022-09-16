"""
Streamlit app for the stroke outcome model. 
"""

# ----- Imports -----
import numpy as np
import matplotlib.pyplot as plt 
import streamlit as st

# For handling the mRS dist files: 
import pandas as pd 
# For highlighting text:
import matplotlib.patheffects as path_effects

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

# Set page to widescreen:
# (must be first call to st)
st.set_page_config(layout='wide')

# 
st.title('Stroke outcome modelling')
st.header('How to use this')
st.write('select stuff on the left sidebar')
# st.subheader('test')

# ----- Fixed parameters ----- 
# Define maximum treatment times:
time_no_effect_ivt = int(6.3*60)   # minutes
time_no_effect_mt = int(8*60)      # minutes

# Utility for each mRS:
utility_weights = np.array([0.97, 0.88, 0.74, 0.55, 0.20, -0.19, 0.00])

# Change default colour scheme:
plt.style.use('seaborn-colorblind')
# Get current matplotlib style sheet colours:
colour_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
# Remove extra colours:
colour_list = colour_list[:7-1]
# Add grey for mRS=6 bin:
colour_list.append('DarkSlateGray')
# Change to array for easier indexing later:
colour_list = np.array(colour_list)


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

# mRS_input = st.sidebar.multiselect(
#     'Choose the pre-stroke mRS value(s) to highlight.',
#     range(6)
#     )
# mRS_input = np.sort(mRS_input)

# if any('MT' in treatment for treatment in treatment_input): 
time_no_effect = (time_no_effect_mt if 'MT' in treatment_input 
            else time_no_effect_ivt)

time_input = st.sidebar.slider(
    'Choose the treatment time in minutes.',
    0, time_no_effect, 60 # start, end, default value
    )
treatment_time_str = (
    f'Treatment time: {time_input//60} hours {time_input%60} minutes')
st.sidebar.write(treatment_time_str)

occlusion_str = 'nlvo' if 'nLVO' in occlusion_input else 'lvo'
treatment_str = 'mt' if 'MT' in treatment_input else 'ivt'
excess_death_str = '_'+treatment_str+'_deaths' 

title = (f'Treating {occlusion_str} with {treatment_str.upper()} ' +
    f'at treatment time {time_input//60} hours {time_input%60} minutes')
st.header(title)


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
except:
    err_str = f'No data for {occlusion_str} and {treatment_str.upper()}.'
    st.warning(err_str)
    st.stop()



# # ----- Use mRS input to define highlight lines -----
# probs_to_mark = []
# for mRS in mRS_input:
#     prob_below = dist_cumsum_pre_stroke[mRS-1] if mRS>0 else 0.0 
#     prob_above = dist_cumsum_pre_stroke[mRS] if mRS<6 else 1.0 
#     probs_to_mark.append(prob_below)
#     probs_to_mark.append(prob_above)


# ----- Find probability with time -----
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


# ----- Add legend -----

# Make dummy columns to force legend to be smaller:
leg_cols = st.columns(3)

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
    ax=ax_probs_time)

ax_probs_time.axvline(time_input/60.0, color='k', linestyle=':')
ax_probs_time.annotate('|',
    xy=(time_input/60.0, 0.0), va='top', ha='center', color='r',
    fontsize=20, zorder=0
    )
ax_probs_time.annotate('\n\nTreatment',
    xy=(time_input/60.0, 0.0), va='top', ha='center', color='r'
    )
# Secret annotation to make sure axis doesn't resize when time input
# is equal to max time:
ax_probs_time.annotate('Treatment',
    xy=(time_no_effect/60.0, 0.0), va='top', ha='center', color='None'
    )

plot_col1.pyplot(fig_probs_time)


# ----- Plot probability distributions ----- 
plot_col2.subheader('The effect of treatment on mRS')

bar_height = 0.5
y_list = [2, 1, 0]
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

dists_to_bar = [
    dist_pre_stroke, 
    dist_time_input_treatment,
    dist_no_treatment
    ]
dists_cumsum_to_bar = [
    dist_cumsum_pre_stroke, 
    dist_cumsum_time_input_treatment,
    dist_cumsum_no_treatment
    ]



(mRS_dist_mix, weighted_added_utils, mRS_list_time_input_treatment, 
    mRS_list_no_treatment) = find_added_utility_between_dists(
    dist_cumsum_time_input_treatment, dist_cumsum_no_treatment,
    utility_weights
)

fig_bars_change, axs = plt.subplots(2, 1, figsize=(8,8))
ax_bars = axs[0] 
ax_util_change = axs[1]


# # Vertical lines for input mRS (pre-stroke) highlights
# for prob in np.append([0.0], dist_cumsum_pre_stroke):
#     ax_bars.axvline(prob, color='grey', linestyle='--', zorder=0)

if plot_col2.checkbox('Mark changes between treated and no treatment'):
    top_of_bottom_bar = y_list[2]+bar_height*0.5
    bottom_of_top_bar = y_list[1]-bar_height*0.5
    for prob in mRS_dist_mix:
        ax_bars.vlines(prob, bottom_of_top_bar, top_of_bottom_bar,
            color='silver', linestyle='-', zorder=0)
        ax_util_change.axvline(prob, color='silver', zorder=0)


for i, dist in enumerate(dists_to_bar):
    draw_horizontal_bar(dist, y=y_list[i], 
        colour_list=colour_list, bar_height=0.5,
        ax=ax_bars)

ax_bars.set_yticks(y_list)
ax_bars.set_yticklabels(y_labels)

# Extend xlims slightly to not cut off bar border colour. 
for ax in axs:
    ax.set_xlim(-5e-3, 1.0+5e-3)
    ax.set_xlabel('Probability')
    ax.set_xticks(np.arange(0,1.01,0.2))
    ax.set_xticks(np.arange(0,1.01,0.05), minor=True)
# ax_bars.tick_params(top=True, bottom=True, which='both')

# Remove sides of the frame:
for spine in ['left', 'right', 'top']:
    ax_bars.spines[spine].set_color(None)


# Bottom subplot
if occlusion_str=='nlvo':
    ylim_util_change = [-0.02, 0.162] # nLVO IVT
else:
    if treatment_str=='mt':
        ylim_util_change = [-0.05, 0.310] # LVO MT
    else:
        ylim_util_change = [-0.027, 0.062] # LVO IVT


if plot_col2.checkbox('Add mRS colours to utility line'):
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
    write_text_about_utility_colours = 1
else:
    write_text_about_utility_colours = 0


ax_util_change.plot(mRS_dist_mix, weighted_added_utils, color='k', 
    label='Cumulative weighted added utility')

ax_util_change.set_ylabel('Cumulative\nweighted added utility')
ax_util_change.set_xlabel('Probability')
ax_util_change.tick_params(top=True, right=True, which='both')

ax_util_change.set_ylim(ylim_util_change)



plot_col2.pyplot(fig_bars_change)

plot_col2.write('When the value of "probability" lands in different mRS bins between the two distributions, the graph has a slope. When the mRS values are the same, the graph is flat.')
if write_text_about_utility_colours>0:
    plot_col2.write('On either side of the cumulative weighted added utility (the solid black line), the two mRS distributions are shown. Immediately above the line is the treated distribution, and immediately below is the "no treatment" distribution.')



# ----- Show metric for +/- mRS and utility -----
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

# Gather the data to display: 
dists_cumsum = [
    dist_cumsum_pre_stroke, 
    dist_cumsum_t0_treatment, 
    dist_cumsum_time_input_treatment,
    dist_cumsum_no_effect, 
    dist_cumsum_no_treatment
    ]  
dists_bins = [
    dist_pre_stroke, 
    dist_t0_treatment, 
    dist_time_input_treatment,
    dist_no_effect, 
    dist_no_treatment
    ]  

# Build data frames: 
df_dists_cumsum = pd.DataFrame(dists_cumsum, 
    index=headings_rows, columns=headings_cols_cumsum)
df_dists_bins = pd.DataFrame(dists_bins, 
    index=headings_rows, columns=headings_cols_bins)

# If box is ticked, display the tables: 
dataframe_expander = st.expander('Show data tables')
with dataframe_expander:
    st.write('Probability bins:')
    st.dataframe(df_dists_bins)

    st.write('Cumulative probability bins:')
    st.dataframe(df_dists_cumsum)