import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from .fixed_params import make_fig_legend, colour_list, utility_weights
from .added_utility_between_dists import find_added_utility_between_dists
from .plot_dist import draw_horizontal_bar


def main(
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
        ):

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
        draw_cumulative_changes(
            nlvo_ivt_case1_dict, case1_time_to_ivt, 'nLVO_IVT_case1')

        st.subheader('Case 2')
        draw_cumulative_changes(
            nlvo_ivt_case2_dict, case2_time_to_ivt, 'nLVO_IVT_case2')

    with tab2:
        # LVO IVT
        st.subheader('The effect of treatment on mRS')
        st.subheader('Case 1')
        draw_cumulative_changes(
            lvo_ivt_case1_dict, case1_time_to_ivt, 'LVO_IVT_case1')

        st.subheader('Case 2')
        draw_cumulative_changes(
            lvo_ivt_case2_dict, case2_time_to_ivt, 'LVO_IVT_case2')

    with tab3:
        # LVO MT
        st.subheader('The effect of treatment on mRS')
        st.subheader('Case 1')
        draw_cumulative_changes(
            lvo_mt_case1_dict, case1_time_to_mt, 'LVO_MT_case1')

        st.subheader('Case 2')
        draw_cumulative_changes(
            lvo_mt_case2_dict, case2_time_to_mt, 'LVO_MT_case2')

    with tab4:
        st.write('Nothing to see here.')


def draw_cumulative_changes(dist_dict, treatment_time, key_str=''):
    """Populate the tab"""
    mRS_dist_mix, weighted_added_utils, \
        mRS_list_time_input_treatment, mRS_list_no_treatment = \
        find_added_utility_between_dists(
            dist_dict['dist_cumsum_time_input_treatment'],
            dist_dict['dist_cumsum_no_treatment']
            )
    do_prob_bars(
        dist_dict,
        mRS_dist_mix,
        weighted_added_utils,
        mRS_list_time_input_treatment,
        mRS_list_no_treatment,
        treatment_time,
        key_str
        )


def do_prob_bars(
        dist_dict,
        mRS_dist_mix, weighted_added_utils,
        mRS_list_time_input_treatment,
        mRS_list_no_treatment, time_input, key_str=''
        ):
    time_input_str = f'{time_input//60}hr {time_input%60}min'
    st.write(
        'We can draw some of the data from the table in the ' +
        '"mRS distributions at the treatment times" section above ' +
        'to create these bar charts of mRS probability distributions:'
        )

    # ----- Plot probability distributions -----
    fig_bars_change, ax_bars = plt.subplots(figsize=(8, 2))

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

    top_of_bottom_bar = y_list[2]+bar_height*0.5
    bottom_of_top_bar = y_list[1]-bar_height*0.5
    for prob in mRS_dist_mix:
        ax_bars.vlines(
            prob, bottom_of_top_bar, top_of_bottom_bar,
            color='silver', linestyle='-', zorder=0
            )

    # Extend xlims slightly to not cut off bar border colour.
    ax_bars.set_xlim(-5e-3, 1.0+5e-3)
    ax_bars.set_xlabel('Probability')
    ax_bars.set_xticks(np.arange(0, 1.01, 0.2))
    ax_bars.set_xticks(np.arange(0, 1.01, 0.05), minor=True)

    st.pyplot(fig_bars_change)

    st.write(
        'The weighted mean utility and mRS is calculated using ' +
        'those regions of the chart where the mRS is different ' +
        'between the "No treatment" and "Treated at ' +
        time_input_str + '" bars.'
        )

    st.write('Sums for the cumulative weighted mRS:')
    big_p_str = build_latex_cumsum_string(
            weighted_added_utils,
            mRS_list_time_input_treatment,
            mRS_list_no_treatment,
            mRS_dist_mix,
            util=False)
    st.latex(big_p_str)

    st.write('Sums for the cumulative weighted utility:')
    big_p_str = build_latex_cumsum_string(
            weighted_added_utils,
            mRS_list_time_input_treatment,
            mRS_list_no_treatment,
            mRS_dist_mix,
            util=True)
    st.latex(big_p_str)

    # Check:
    # st.write(weighted_added_utils[-1])


def plot_bars(
        dists_to_bar,
        dists_cumsum_to_bar,
        ax_bars,
        time_input,
        y_list,
        bar_height
        ):
    y_labels = [
        'Pre-stroke',
        ('Treated at \n' + f'{time_input//60}hr ' +
         f'{time_input%60:02d}min'),
        'No treatment'
        ]
    # ^ keep formatting for e.g. 01 minutes in the middle bar
    # otherwise the axis jumps about as the label changes size
    # between "9 minutes" and "10 minutes" (extra character).

    for i, dist in enumerate(dists_to_bar):
        draw_horizontal_bar(
            dist, y=y_list[i],
            colour_list=colour_list, bar_height=0.5,
            ax=ax_bars
            )

    ax_bars.set_yticks(y_list)
    ax_bars.set_yticklabels(y_labels)

    # Remove sides of the frame:
    for spine in ['left', 'right', 'top']:
        ax_bars.spines[spine].set_color(None)


def build_latex_cumsum_string(
        weighted_added_utils,
        mRS_list_time_input_treatment,
        mRS_list_no_treatment,
        mRS_dist_mix,
        util=True):
    cumulative_changes = 0.0
    big_p_str = r'''\begin{align*}'''
    # Add column headings:
    big_p_str += (
        r'''& \mathrm{Treated} & & \mathrm{Not\ treated} ''' +
        r'''& \mathrm{Proportion} \\'''
        )
    for i in range(1, len(weighted_added_utils)):
        if weighted_added_utils[i] - weighted_added_utils[i-1] != 0:

            bin_width = mRS_dist_mix[i]-mRS_dist_mix[i-1]
            # Add a tiny amount to prevent round-to-nearest-even
            # oddities when printing the values.
            bin_width += 1e-7
            p_str_bin_width = f'{bin_width:6.3f}'

            if util is True:
                value_treated = \
                    utility_weights[mRS_list_time_input_treatment[i]]
                value_no_treatment = utility_weights[mRS_list_no_treatment[i]]

                p_str_treated = f'{value_treated:5.2f}'
                p_str_no_treatment = f'{value_no_treatment:5.2f}'

            else:
                value_treated = mRS_list_time_input_treatment[i]
                value_no_treatment = mRS_list_no_treatment[i]

                p_str_treated = f'{value_treated:1d}'
                p_str_no_treatment = f'{value_no_treatment:1d}'

            p_str = ''

            # First weight:
            p_str += r'''(&\textcolor{'''
            p_str += f'{colour_list[mRS_list_time_input_treatment[i]]}'
            p_str += r'''}{'''
            if value_treated >= 0:
                # Add sneaky + for alignment
                p_str += r'\phantom{+}'
            p_str += p_str_treated
            p_str += r'''}&-&\textcolor{'''

            # Second weight:
            p_str += f'{colour_list[mRS_list_no_treatment[i]]}'
            p_str += r'''}{'''
            if value_no_treatment >= 0:
                # Add sneaky + for alignment
                p_str += r'\phantom{+}'
            p_str += p_str_no_treatment
            p_str += r'''} )& \times '''
            # Bin widths:
            p_str += p_str_bin_width
            # Value of this line:
            p_str += r''' = &'''
            value_here = (
                (value_treated - value_no_treatment) * bin_width
            )
            # Round the total of the line to the same number of digits
            # as the printed total of the column, otherwise the sums
            # look incorrect.
            if util is True:
                value_here = round(value_here, 3)
            else:
                value_here = round(value_here, 2)
            cumulative_changes += value_here

            if value_here >= 0:
                # Add sneaky + for alignment
                p_str += r'\phantom{+}'
            if util is True:
                p_str += f'{value_here:6.3f}'
            else:
                p_str += f'{value_here:5.2f}'
            # Next line:
            p_str += '\\\\'

            # Add to the big string:
            big_p_str += p_str

    # Add total beneath the rest:
    # big_p_str += r'''& & & & & & -----\\\\'''
    big_p_str += r'''\hline'''
    big_p_str += r'''& & & & \mathrm{Total}: &'''

    if cumulative_changes >= 0:
        # Add sneaky + for alignment
        big_p_str += r'\phantom{+}'
    if util is True:
        big_p_str += f'{cumulative_changes:6.3f}\\\\'
    else:
        big_p_str += f'{cumulative_changes:5.2f}\\\\'
    big_p_str += r'''\end{align*}'''

    return big_p_str
