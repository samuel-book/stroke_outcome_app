"""
Functions to plot mRS probability distribution varition with time
and to show tables of some important values.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .fixed_params import colour_list, time_no_effect_mt


def plot_probs_with_time(dict1, dict2):
    """
    This function is a wrapper for do_probs_with_time
    and exists to reduce clutter in the main demo script.
    """
    # Plot probs with time
    do_probs_with_time(
        dict1['time_no_effect'],
        dict1['A_list'],
        dict2['b_list'],
        colour_list,
        [dict1['treatment_time'], dict2['treatment_time']],
        treatment_labels=[f'Case {i+1}' for i in range(2)],
        time_no_effect_mt=time_no_effect_mt
        )


def do_probs_with_time(
        time_no_effect,
        A_list,
        b_list,
        colour_list,
        treatment_times,
        treatment_labels=[],
        time_no_effect_mt=8*60
        ):
    """
    Plot probability variation with time.

    For the x-axis hover labels to show a nicely-formatted time,
    plot a datetime object (which shows as e.g. 01:10)
    instead of a list of hours or minutes (which shows as 1.16666...).
    """
    # ----- Define times -----
    # Define times to show probability variation.
    # Plot every five minutes up to the time of no effect:
    times_to_plot = np.arange(0, time_no_effect + 1e-7, 5)
    # Add in the treatment times and time of no effect:
    times_to_plot = np.sort(np.unique(np.append(
        [*treatment_times, time_no_effect], times_to_plot
    )))
    # Convert to datetime object for nicer formatting:
    times_to_plot_m = pd.to_datetime(times_to_plot, unit='m')

    # ----- Define probabilities -----
    # P(mRS<=5)=1.0 at all times, so it has no defined A, a, and b.
    # Instead append to this array a 0.0, which won't be used directly
    # but will allow the "for" loop to go round one extra time.
    A_list = np.append(A_list, 0.0)
    cum_probs_with_time_lists = []
    for i, A_i in enumerate(A_list):
        # --- Cumulative probs for hover label
        # Define the probability line, p_i:
        if i < 6:
            p_i = 1.0/(1.0 + np.exp(-A_i - b_list[i]*times_to_plot))
        else:
            # P(mRS<=5)=1.0 at all times:
            p_i = np.full(times_to_plot.shape, 1.0)
        cum_probs_with_time_lists.append(p_i)

        # --- Non-cumulative probs for direct plotting
        # Convert cumulative hazard lists into non-cumulative
        # for easier plotting with plotly.
        if i < 1:
            probs_with_time_lists = [cum_probs_with_time_lists[0]]
        else:
            # For each mRS, subtract the values that came before it.
            diff_list = np.array(
                cum_probs_with_time_lists[i] -
                cum_probs_with_time_lists[i-1]
                )
            probs_with_time_lists.append(diff_list)

    # ----- Plot -----
    fig = go.Figure()
    # Plot each mRS band in a separate trace:
    for i in range(7):
        # Set up "custom data" for the hover label.
        # Have to use np.stack() because plotly expects to receive
        # this information in columns, not in rows.
        customdata = np.stack((
            cum_probs_with_time_lists[i],
            [i]*len(times_to_plot_m),
            ), axis=-1)
        # Line and fill:
        fig.add_trace(go.Scatter(
            x=times_to_plot_m,
            y=probs_with_time_lists[i],
            mode='lines',
            line=dict(color=colour_list[i]),
            stackgroup='one',    # Stack all traces on this group.
            name=f'{i}',         # Label for the legend is mRS.
            customdata=customdata
        ))
    # The custom_data aren't directly plotted in the previous lines,
    # but are loaded ready for use with the hover template later.

    # Set axis labels:
    fig.update_xaxes(title_text='Time since onset')
    fig.update_yaxes(title_text='Cumulative probability')
    fig.update_layout(legend_title='mRS')
    # Format x-axis tick labels to show hours (%H) and minutes (%M):
    fig.update_xaxes(tickformat="%Hh %Mm")

    # Set axis limits:
    # Make the y-axis max a bit bigger than 1 to make sure the label
    # at 1 is shown.
    fig.update_yaxes(range=[0, 1 + 1e-2])
    # Convert x limits to datetime to match the plotted data:
    x_min = pd.to_datetime(0.0, unit='m')
    # Give breating room to allow an empty space with no hover label
    # so that on touch devices there's a space to touch that removes
    # the current label.
    x_max = pd.to_datetime(time_no_effect_mt + 60, unit='m')
    fig.update_xaxes(range=[x_min, x_max], constrain='domain')

    # Hover settings:
    # When hovering, highlight all mRS bins' points for chosen x:
    fig.update_layout(hovermode='x unified')
    # Show this message when hovering:
    fig.update_traces(
        hovertemplate=(
            # mRS ≤ {mRS}: {cumulative probability}
            'mRS≤%{customdata[1]}: %{customdata[0]:6.4f}' +
            '<extra></extra>'
            )
    )
    # The line with <extra></extra> is required
    # to remove the "secondary box" hover label before the rest of the
    # hover template. Otherwise get "0 mRS=0 ..."

    # Add vertical line at treatment times.
    for i, treatment_time in enumerate(treatment_times):
        # Convert the time to datetime object to match plotted x data:
        treatment_time_m = pd.to_datetime(treatment_time, unit='m')
        fig.add_vline(
            x=treatment_time_m,
            line=dict(color='black', width=2.0)
            )
        fig.add_annotation(
            x=treatment_time_m,
            y=1.0,
            text=treatment_labels[i],
            showarrow=True,
            arrowhead=0,
            ax=0,    # Make arrow vertical - a = arrow, x = x-shift.
            ay=-20,  # Make the label sit above the top of the graph
            textangle=-45
            )

    # Format legend:
    fig.update_layout(legend=dict(
        orientation='h',      # horizontal
        # traceorder='normal',  # Show mRS=0 on left,
        #                       # but also mRS=0 on bottom on hover :(
        # Location:
        x=1.0,
        y=1.3,
        yanchor='bottom',
        xanchor='right',
        # Remove interactive legend (clicking to highlight or hide):
        itemclick=False,
        itemdoubleclick=False,
        # Fiddle with size of each entry to change legend width
        entrywidth=2,
        ))

    # Remove grid lines:
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    # Set aspect ratio:
    fig.update_yaxes(
        scaleanchor='x',
        scaleratio=2.5,
        constrain='domain'
    )
    # Reduce size of figure by adjusting margins:
    fig.update_layout(margin=dict(b=0, t=30), height=300)

    # Disable zoom and pan:
    fig.update_layout(xaxis=dict(fixedrange=True),
                      yaxis=dict(fixedrange=True))

    # Turn off legend click events
    # (default is click on legend item, remove that item from the plot)
    fig.update_layout(legend_itemclick=False)

    # Options for the mode bar.
    # (which doesn't appear on touch devices.)
    plotly_config = {
        # Mode bar always visible:
        # 'displayModeBar': True,
        # Plotly logo in the mode bar:
        'displaylogo': False,
        # Remove the following from the mode bar:
        'modeBarButtonsToRemove': [
            'zoom', 'pan', 'select', 'zoomIn', 'zoomOut', 'autoScale',
            'lasso2d'
            ],
        # Options when the image is saved:
        'toImageButtonOptions': {'height': None, 'width': None},
        }

    # Write to streamlit:
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)


def table_probs_with_time(dict1, dict2):
    # # Tabulate mRS bins
    st.markdown('''
        The following table contains the probability distributions
        at key points.
        '''
        )
    df_combo = make_combo_mRS_bin_dataframe(
        dict1['df_dists_bins'], dict2['df_dists_bins'],
        dict1['treatment_time'], dict2['treatment_time'])
    st.dataframe(df_combo, use_container_width=True)


def make_combo_mRS_bin_dataframe(df1, df2, treatment_time1, treatment_time2):
    # Find whether Case 1 or Case 2 has the smaller treatment time.
    # Use the smaller one as the main dataframe,
    # and the larger for an extra row.
    if treatment_time1 < treatment_time2:
        df_main = df1
        df_extra = df2
    elif treatment_time2 < treatment_time1:
        df_main = df2
        df_extra = df1
    else:
        # Same rows in both so just return one:
        return df1

    new_df = pd.concat((
        df_main.iloc[:3],    # Pre-stroke, time=0, time=first treatment time,
        df_extra.iloc[2:3],  # time=second treatment time,
        df_main.iloc[3:]     # time=no-effect, not treated.
        ), axis=0)
    return new_df
