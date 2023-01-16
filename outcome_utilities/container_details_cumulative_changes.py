import streamlit as st
import numpy as np
import plotly.graph_objects as go

from .fixed_params import colour_list, utility_weights
from .added_utility_between_dists import find_added_utility_between_dists
# from .plot_dist import draw_horizontal_bar


def draw_cumulative_changes(dist_dict, treatment_time, key_str=''):
    """Populate the tab"""
    mRS_dist_mix, weighted_added_utils, \
        mRS_list_time_input_treatment, mRS_list_no_treatment = \
        find_added_utility_between_dists(
            dist_dict['dist_cumsum_time_input_treatment'],
            dist_dict['dist_cumsum_no_treatment']
            )
    st.write(
        'We can draw some of the data from the table in the ' +
        '"mRS distributions at the treatment times" section above ' +
        'to create these bar charts of mRS probability distributions:'
        )
    do_prob_bars(dist_dict, treatment_time)
    time_input_str = f'{treatment_time//60}hr {treatment_time%60}min'
    st.write(
        'The weighted mean utility and mRS is calculated using ' +
        'those regions of the chart where the mRS is different ' +
        'between the "No treatment" and "Treated at ' +
        time_input_str + '" bars.'
        )
    write_latex_sums_for_weighted_mRS(
            weighted_added_utils,
            mRS_list_time_input_treatment,
            mRS_list_no_treatment,
            mRS_dist_mix
            )


def do_prob_bars(dist_dict, time_input):
    # Get the information from these keys in the input dist_dict:
    dists = [
        'dist_no_treatment',
        'dist_time_input_treatment',
        'dist_pre_stroke'
        ]
    cum_dists = [
        'dist_cumsum_no_treatment',
        'dist_cumsum_time_input_treatment',
        'dist_cumsum_pre_stroke'
        ]
    # Choose labels for the bars:
    y_labels = [
        'No treatment',
        'Treated at\n' + f'{time_input//60}hr ' + f'{time_input%60:02d}min',
        'Pre-stroke'
        ]
    # ^ keep label formatting for e.g. 01 minutes in the middle bar
    # otherwise the axis jumps about as the label changes size
    # between "9 minutes" and "10 minutes" (extra character).

    # Place the bars at these locations:
    y_vals = [0, 1, 2]

    fig = go.Figure()
    # Add the stacked bar for each distribution as a separate trace.
    for i, dist in enumerate(dists):
        # Only add the mRS colours to the legend if this is the
        # first go round the loop.
        show_legend = False if i > 0 else True
        # Draw each bar individually to pick the colours we want:
        for mRS in np.arange(7):
            # Put mRS label and cumulative probability value in
            # custom data array for use in the hover label:
            custom_data = np.stack((
                [mRS],
                [dist_dict[cum_dists[i]][mRS]],
            ), axis=-1)
            # Draw the bar:
            fig.add_trace(go.Bar(
                x=[dist_dict[dist][mRS]],
                y=[y_vals[i]],
                marker=dict(color=colour_list[mRS]),
                orientation='h',         # horizontal
                name=str(mRS),           # name for legend
                showlegend=show_legend,  # True/False appear in legend
                customdata=custom_data,
                width=0.7                # Skinniness of bars
            ))
    # The custom_data aren't directly plotted in the previous lines,
    # but are loaded ready for use with the hover template later.

    # Change the bar mode
    fig.update_layout(barmode='stack')

    # Update message displayed on hover:
    fig.update_traces(
        hovertemplate=(
            'mRS≤%{customdata[0]}: %{customdata[1]:.3f}' +
            '<br>' +               # (line break)
            'mRS=%{customdata[0]}: %{x:.3f}' +
            '<extra></extra>'      # Remove content from secondary box.
            )
    )

    # Set axis labels:
    fig.update_xaxes(title_text='Cumulative probability')
    fig.update_layout(legend_title='mRS')
    # Y tick labels:
    fig.update_layout(yaxis=dict(
        tickmode='array',
        tickvals=y_vals,
        ticktext=y_labels
    ))

    # Format legend:
    fig.update_layout(legend=dict(
        orientation='h',      # horizontal
        traceorder='normal',  # Show mRS=0 on left
        # Location:
        x=1.0,
        y=1.3,
        yanchor='bottom',
        xanchor='right',
        # Remove interactive legend (clicking to highlight or hide):
        itemclick=False,
        itemdoubleclick=False
        ))

    # Set axis limits:
    # (give breathing room for the bar borders to be drawn)
    fig.update_xaxes(range=[0 - 1e-2, 1 + 1e-2])
    fig.update_yaxes(range=[min(y_vals) - 0.4, max(y_vals) + 0.4])

    # Remove grid lines and x=0, y=0 lines:
    fig.update_xaxes(zeroline=False, showgrid=False)
    fig.update_yaxes(zeroline=False, showgrid=False)

    # Make plot less tall:
    fig.update_layout(margin=dict(t=50, b=0), height=150)

    # Write to streamlit:
    st.plotly_chart(fig, use_container_width=True)


def write_latex_sums_for_weighted_mRS(
        weighted_added_utils,
        mRS_list_time_input_treatment,
        mRS_list_no_treatment,
        mRS_dist_mix
        ):
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
