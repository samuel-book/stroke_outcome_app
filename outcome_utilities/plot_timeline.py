"""
Plots a timeline for the patient pathway for cases 1 and 2.
Includes labelled points and emoji.
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from outcome_utilities.fixed_params import emoji_text_dict, plotly_colours


def make_timeline_plot(time_dicts):
    label_dict = dict(
        onset='Onset',
        onset_to_ambulance_arrival='Travel to<br>hospital begins',
        travel_to_ivt='Arrive at<br>IVT centre',
        travel_to_mt='Arrive at<br>IVT+MT centre',
        ivt_arrival_to_treatment='IVT',
        transfer_additional_delay='Transfer begins',
        travel_ivt_to_mt='Arrive at<br>IVT+MT centre',
        mt_arrival_to_treatment='MT',
        )

    # Set up x values for the timelines, labels, and emoji:
    x_vals = [0, 1.2]
    label_offset = 0.6
    emoji_offset = 0.0
    # For the y-axis limit, set up y_max and update it with the largest
    # cumulative time to plot.
    y_max = 0.0

    fig = go.Figure()
    for i, time_dict in enumerate(time_dicts):
        # Draw the timeline at this x-value:
        x = x_vals[i]

        # Loop over the time dictionary and fill these lists with
        # information for plotting:
        labels = []                  # Name of checkpoint.
        time_cum_list = []           # Cumulative times.
        time_cum_str_list = []       # same, as formatted strings.
        time_colour_list = []        # Colour of the time.
        emoji_to_draw = []           # Emoji for each checkpoint.
        # Whether to write formatted time string under the label:
        write_under_list = []        # Each value is True/False

        # Keep track of the cumulative time to each checkpoint:
        time_cumulative = 0.0
        for time_key in time_dict.keys():
            # Store how to label this on the plot:
            labels.append(label_dict[time_key])
            # Find the emoji to draw next to this label.
            # If there isn't one (e.g. for "onset"), an empty string
            # will be drawn instead.
            try:
                emoji_to_draw.append(emoji_text_dict[time_key])
            except KeyError:
                emoji_to_draw.append('')
            # Find the cumulative time to this point:
            t_min = time_dict[time_key]       # minutes
            time_cumulative += t_min/60.0     # hours
            time_cum_list.append(time_cumulative)
            # Convert value to string for plotting:
            time_cum_str = (
                f'{int(60*time_cumulative//60):2d}hr ' +
                f'{int(60*time_cumulative%60):2d}min'
                )
            time_cum_str_list.append(time_cum_str)

            # Decide colour formatting and extra labels for special
            # cases:
            if 'ivt_arrival_to_treatment' in time_key:
                colour = plotly_colours[0]
                write_under = True    # Write formatted time string
            elif 'mt_arrival_to_treatment' in time_key:
                colour = plotly_colours[1]
                write_under = True    # Write formatted time string
            else:
                # Default colour
                colour = None         # black/white in light/dark mode
                write_under = False   # Don't write time string.
            time_colour_list.append(colour)
            write_under_list.append(write_under)

        # --- Plot ---
        # Make new labels list with line breaks removed
        # (for use in the hover label):
        labels_plain = [l.replace('<br>', ' ') for l in labels]
        # Draw straight line along the time axis:
        fig.add_trace(go.Scatter(
            y=time_cum_list,
            x=[x] * len(time_cum_list),
            mode='lines+markers',
            marker=dict(size=6, symbol='line-ew-open'),
            line=dict(color='grey'),    # OK in light and dark mode.
            showlegend=False,
            customdata=np.stack((time_cum_str_list, labels_plain), axis=-1)
        ))
        # "customdata" is not directly plotted in the line above,
        # but the values are made available for the hover label.

        # Update the hover text for the lines:
        fig.update_traces(
            hovertemplate=(
                '%{customdata[1]}'          # Name of this checkpoint
                '<br>' +                    # (line break)
                'Time: %{customdata[0]}' +  # Formatted time string
                '<extra></extra>'           # Remove secondary box.
                )
            )

        # Add label for each scatter marker
        for t, time_cum in enumerate(time_cum_list):
            # Only show it if it's moved on from the previous:
            if t == 0 or time_cum_list[t] > time_cum_list[t-1]:
                if write_under_list[t] is True:
                    # Add formatted time string to the label.
                    # (plus a line break, <br>)
                    text = labels[t] + '<br>'+time_cum_str_list[t]
                else:
                    text = labels[t]
                # Write the label:
                fig.add_annotation(
                    y=time_cum,
                    x=x + label_offset,
                    text=text,
                    showarrow=False,
                    font=dict(color=time_colour_list[t], size=10),
                    )
                # Add emoji for each scatter marker
                fig.add_annotation(
                    y=time_cum,
                    x=x + emoji_offset,
                    text=emoji_to_draw[t],
                    showarrow=False,
                    font=dict(color=time_colour_list[t])
                    )

            # Update y-axis limit value if necessary:
            if time_cumulative > y_max:
                y_max = time_cumulative

    # Set y range:
    fig.update_yaxes(range=[y_max * 1.05, 0 - y_max * 0.025])
    # Set y-axis label
    fig.update_yaxes(title_text='Time since onset (hours)')
    # Change y-axis title font size:
    fig.update_yaxes(title_font_size=10)

    # Set x range:
    fig.update_xaxes(range=[-0.1, 2.5])
    # Set x-axis labels
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=x_vals,
            ticktext=['<b>Case 1', '<b>Case 2'],   # <b> for bold
            side='top'  # Moves the labels to the top of the grid
        ),
    )

    # Remove y=0 and x=0 lines (zeroline) and grid lines:
    fig.update_yaxes(zeroline=False, showgrid=False)
    fig.update_xaxes(zeroline=False, showgrid=False)


    # Reduce size of figure by adjusting margins:
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=500)

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