"""
Streamlit app for the stroke outcome model. 
"""

# ----- Imports -----
from re import A
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
    'Choose the pre-stroke mRS value(s) to highlight.',
    range(6)
    )
mRS_input = np.sort(mRS_input)

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
    err_str = f'No data for {occlusion_str} and {treatment_str}.'
    st.write(err_str)

    # Provide some dummy data so the rest of the script runs: 
    dist_dummy = np.array([np.NaN for i in range(7)])
    dist_pre_stroke = dist_dummy
    dist_no_treatment = dist_dummy
    dist_t0_treatment = dist_dummy
    dist_no_effect = dist_dummy

    dist_cumsum_pre_stroke = dist_dummy
    dist_cumsum_no_treatment = dist_dummy
    dist_cumsum_t0_treatment = dist_dummy
    dist_cumsum_no_effect = dist_dummy


# ----- Use mRS input to define highlight lines -----
probs_to_mark = []
for mRS in mRS_input:
    prob_below = dist_cumsum_pre_stroke[mRS-1] if mRS>0 else 0.0 
    prob_above = dist_cumsum_pre_stroke[mRS] if mRS<6 else 1.0 
    probs_to_mark.append(prob_below)
    probs_to_mark.append(prob_above)


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


# --- Plot probability with time -----
st.write('Probability distribution with time:')
times_to_plot = np.linspace(0, time_no_effect, 20)

fig_probs_time, ax_probs_time = plt.subplots()
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

st.pyplot(fig_probs_time)


# ----- Plot probability distributions ----- 
st.write('Compare the probability distribution at this treatment time:')
fig_bars, ax_bars = plt.subplots(gridspec_kw={'left':0.1, 'right':0.9}) 

bar_height = 0.5
y_list = [2, 1, 0]
y_labels = [
    'Pre-stroke', 
    (f'Treated with {treatment_str} at '+'\n'
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
for i, dist in enumerate(dists_to_bar):
    draw_horizontal_bar(dist, y=y_list[i], 
        colour_list=colour_list, bar_height=0.5,
        ax=ax_bars)

# Vertical lines for input mRS (pre-stroke) highlights
# for prob in probs_to_mark:
for prob in np.append([0.0], dist_cumsum_pre_stroke):
    ax_bars.axvline(prob, color='grey', linestyle='--', zorder=0)

# # Within highlighted area, annotate the bar sizes. 
# for mRS in mRS_input:
#     pre_stroke_bin_size = dist_pre_stroke[mRS]
#     pre_stroke_bin_left = dist_cumsum_pre_stroke[mRS-1] if mRS>0 else 0.0
#     pre_stroke_bin_right = dist_cumsum_pre_stroke[mRS] if mRS<5 else 1.0

#     # Annotate the pre-stroke mRS value:
#     pre_stroke_bin_mid = np.mean([pre_stroke_bin_left, pre_stroke_bin_right])
#     ax_bars.annotate(f'mRS\n{mRS}', 
#         xy=(pre_stroke_bin_mid, y_list[0]+0.5*bar_height),
#         color=colour_list[mRS], 
#         ha='center', va='bottom')

#     # Find which mRS bins here are within the highlight: 
#     # Middle bar:
#     time_input_bin_smallest = np.where(
#         dist_cumsum_time_input_treatment>=pre_stroke_bin_left)[0][0]
#     time_input_bin_largest = np.where(
#         dist_cumsum_time_input_treatment>=pre_stroke_bin_right)[0][0]
#     # Bottom bar:
#     no_treatment_bin_smallest = np.where(
#         dist_cumsum_no_treatment>=pre_stroke_bin_left)[0][0]
#     no_treatment_bin_largest = np.where(
#         dist_cumsum_no_treatment>=pre_stroke_bin_right)[0][0]


(mRS_dist_mix, weighted_added_utils, mRS_list_time_input_treatment, 
    mRS_list_no_treatment) = find_added_utility_between_dists(
    dist_cumsum_time_input_treatment, dist_cumsum_no_treatment,
    utility_weights
)

st.write(mRS_dist_mix)
st.write(weighted_added_utils)

y_a = y_list[1]-bar_height*0.5
y_b = y_list[2]+bar_height*0.5
y_mid = np.mean([y_list[1], y_list[2]])
for i in range(len(mRS_list_time_input_treatment)):
    mRS_a = mRS_list_time_input_treatment[i]
    mRS_b = mRS_list_no_treatment[i]
    # p_a = dist_cumsum_time_input_treatment[mRS_a]
    # p_a2 = dist_cumsum_time_input_treatment[mRS_a-1]
    # p_b = dist_cumsum_no_treatment[mRS_b]
    # p_b = dist_cumsum_no_treatment[mRS_b-1]
    p_a = mRS_dist_mix[i] 
    p_b = mRS_dist_mix[i-1] if i>0 else 0.0
    if mRS_a!=mRS_b:
        # ax_bars.vlines(mRS_dist_mix[i], 
        #     y_list[1]+bar_height*0.5,
        #     y_list[2]-bar_height*0.5, 
        #     color='LimeGreen', zorder=0)
        ax_bars.fill(
            [p_a,p_a,p_b,p_b],
            [y_mid,y_b,y_b,y_mid],
            color=colour_list[mRS_b],#'Gainsboro', 
            # edgecolor=colour_list[mRS_a],#'grey',
            # hatch='---',
            edgecolor='None',
            alpha=0.4,
            zorder=0
            )
        ax_bars.fill(
        [p_a,p_a,p_b,p_b],
        [y_mid,y_a,y_a,y_mid],
        color=colour_list[mRS_a],#'Gainsboro', 
        # edgecolor=colour_list[mRS_a],#'grey',
        # hatch='---',
        edgecolor='None',
        alpha=0.4,
        zorder=0
        )
        ax_bars.vlines(p_a, y_a, y_b, color='k', linewidth=0.5)
        ax_bars.vlines(p_b, y_a, y_b, color='k', linewidth=0.5)


ax_bars.set_yticks(y_list)
ax_bars.set_yticklabels(y_labels)

ax_bars.set_xlabel('Probability')
ax_bars.set_xticks(np.arange(0,1.01,0.2))
ax_bars.set_xticks(np.arange(0,1.01,0.05), minor=True)
# Extend xlims slightly to not cut off bar border colour. 
ax_bars.set_xlim(-5e-3, 1.0+5e-3)
# ax_bars.tick_params(top=True, bottom=True, which='both')

# Remove sides of the frame:
for spine in ['left', 'right', 'top']:
    ax_bars.spines[spine].set_color(None)

st.pyplot(fig_bars)


# ----- Show metric for +/- mRS and utility -----
# Calculate mean mRSes:
mean_mRS_no_treatment = np.mean(dist_no_treatment*np.arange(7))
mean_mRS_time_input_treatment = np.mean(
    dist_time_input_treatment*np.arange(7))
mean_mRS_diff = mean_mRS_time_input_treatment - mean_mRS_no_treatment

# Calculate mean utilities: 
# (it seems weird to use "sum" instead of "mean" but this definition 
# matches the clinical outcome script)
mean_utility_no_treatment = np.sum(dist_no_treatment*utility_weights)
mean_utility_time_input_treatment = np.sum(
    dist_time_input_treatment*utility_weights)
mean_utility_diff = (
    mean_utility_time_input_treatment - mean_utility_no_treatment)

# Put the two metrics in columns: 
col1, col2 = st.columns(2)
col1.metric('Population mean mRS', 
    f'{mean_mRS_time_input_treatment:0.3f}', 
    f'{mean_mRS_diff:0.3f} from "no treatment"',
    delta_color='inverse' # A negative difference is good.
    )
col2.metric('Population mean utility', 
    f'{mean_utility_time_input_treatment:0.3f}', 
    f'{mean_utility_diff:0.3f} from "no treatment"',
    )


# ----- Show data frame -----
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
if st.checkbox('Show data tables'):
    st.write('Probability bins:')
    st.dataframe(df_dists_bins)

    st.write('Cumulative probability bins:')
    st.dataframe(df_dists_cumsum)