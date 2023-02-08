"""
Plots a timeline for the patient pathway for cases 1 and 2.
Includes labelled points and emoji.
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from outcome_utilities.fixed_params import emoji_text_dict, plotly_colours


def make_timeline_plot(time_dicts, input_type='Pathway'):
    if input_type == 'Pathway':
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
    else:
        label_dict = dict(
            onset='Onset',
            time_to_ivt='IVT',
            time_to_mt='MT'
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
            if input_type == 'Pathway':
                time_cumulative += t_min/60.0     # hours
            else:
                time_cumulative = t_min / 60.0
                # Update y-axis limit value if necessary:
                if time_cumulative > y_max:
                    y_max = time_cumulative
            time_cum_list.append(time_cumulative)
            # Convert value to string for plotting:
            time_cum_str = (
                f'{int(60*time_cumulative//60):2d}hr ' +
                f'{int(60*time_cumulative%60):2d}min'
                )
            time_cum_str_list.append(time_cum_str)

            # Decide colour formatting and extra labels for special
            # cases:
            if 'ivt_arrival_to_treatment' in time_key or 'time_to_ivt' in time_key:
                colour = plotly_colours[0]
                write_under = True    # Write formatted time string
            elif 'mt_arrival_to_treatment' in time_key or 'time_to_mt' in time_key:
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
            if t == 0 or time_cum_list[t] > time_cum_list[t-1] or input_type == 'Simple':
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


def plot_simple_map(travel_to_ivt, travel_ivt_to_mt, travel_to_mt):
    # Emoji for the centres:
    emoji = ['', '\U0001f3e5', '\U0001f3e5']
    # Labels for the centres:
    labels = [
        'Onset<br>location',
        'IVT<br>centre',
        'IVT+MT<br>centre'
        ]
    # Labels for the lines linking the centres:
    labels_link = [
        f'{travel_to_ivt}' + '<br> \U0001f691',
        '\U0001f691 <br>' + f'{travel_ivt_to_mt}',
        '\U0001f691 <br>' + f'{travel_to_mt}',
        ]

    # Work out the coordinates.
    coords_onset = [0.0, 0.0]
    coords_ivt = [travel_to_ivt, 0.0]
    # Find the crossover between the remaining two distances,
    # travel to mt from onset and
    # travel from ivt to mt.
    # Use cosine rule to find the angle we need:
    try:
        th = np.arccos(
            (travel_to_ivt**2.0 + travel_to_mt**2.0 - travel_ivt_to_mt**2.0) /
            (2 * travel_to_ivt * travel_to_mt)
            )
        if np.isnan(th) == True:
            # ^ linter complains, but it breaks if you change this
            #   == to "is".
            # Can't draw this map!
            st.caption('''
                (Sorry - the requested travel times between hospitals
                are impossible to draw on this simple grid.)
                ''')
            return
        else:
            # Use this to generate the coordinates:
            coords_mt = [travel_to_mt * np.cos(th), -travel_to_mt * np.sin(th)]
        # The length of the horizontal line:
        h_travel = travel_to_ivt
    except ZeroDivisionError:
        # Either travel to IVT or to MT is zero.
        # So remove the label for the onset location:
        labels[0] = ''
        labels_link[1] = ''
        # Set the coordinates along a straight line
        # and remove any labels along the link that doesn't exist anymore.
        if travel_to_ivt == 0.0:
            coords_ivt = [0.0, 0.0]
            labels_link[0] = ''
            coords_mt = [travel_to_mt, 0.0]
            # The length of the horizontal line:
            h_travel = travel_to_mt
        else:
            coords_mt = [0.0, 0.0]
            labels_link[2] = ''
            # The length of the horizontal line:
            h_travel = travel_to_ivt
        # Set some angle for scaling the label offsets later:
        th = np.pi * 0.5

    all_coords = [coords_onset, coords_ivt, coords_mt]

    # Use coords to define axis limits.
    x_min = np.min(np.array(all_coords).ravel()[::2])
    x_max = np.max(np.array(all_coords).ravel()[::2])
    y_max = np.max(np.array(all_coords).ravel()[1::2])
    y_min = np.min(np.array(all_coords).ravel()[1::2])
    # Find the extent of the grid for label placement:
    x_span = x_max - x_min
    y_span = y_max - y_min

    # Set label placements:
    label_offset = y_span*0.2
    # If this offset is small compared with the width of the plot,
    # reset it to something larger:
    if label_offset < 0.15*h_travel:
        label_offset = 0.15*h_travel
    label_y_offsets = [label_offset, label_offset, -label_offset]

    fig = go.Figure()

    for i, coords in enumerate(all_coords):
        # Draw marker
        fig.add_trace(go.Scatter(
            x=[coords[0]],
            y=[coords[1]],
            mode='markers',
            marker=dict(symbol='circle', color='grey'),
            showlegend=False,
            hoverinfo='skip'
        ))
        # Add label for each treatment centre
        fig.add_annotation(
            x=coords[0],
            y=coords[1] + label_y_offsets[i],
            text=labels[i],
            showarrow=False
            )

        # Add emoji for each treatment centre
        fig.add_annotation(
            x=coords[0],
            y=coords[1],
            text=emoji[i],
            showarrow=False
            )

    # Draw links between the centres.
    # Setup for the lines:
    coords_links = [
        [coords_onset, coords_ivt],
        [coords_ivt, coords_mt],
        [coords_onset, coords_mt]
    ]
    colours_links = [plotly_colours[0], plotly_colours[0], plotly_colours[1]]
    linestyles_links = ['solid', 'solid', 'dot']

    # Setup for the text labels:
    label_link_x_offsets = [
        0,
        label_offset * np.sin(th) * 0.5,
        -label_offset * np.sin(th) * 0.5
        ]
    label_link_y_offsets = [
        label_offset * 0.5,
        0.0,
        0.0
        ]

    for i, coords in enumerate(coords_links):
        x = [coords[0][0], coords[1][0]]
        y = [coords[0][1], coords[1][1]]
        # Draw connections:
        fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                line=dict(color=colours_links[i], dash=linestyles_links[i]),
                showlegend=False,
                hoverinfo='skip'
            ))
        # Find x and y coords halfway along this line:
        x_halfway = np.mean(x)
        y_halfway = np.mean(y)
        # Draw annotation of distance:
        fig.add_annotation(
            x=x_halfway + label_link_x_offsets[i],
            y=y_halfway + label_link_y_offsets[i],
            text=labels_link[i],
            font=dict(color=colours_links[i]),
            showarrow=False
            )


    # Remove x and y axis ticks:
    fig.update_xaxes(dict(
        tickmode='array',
        tickvals=[],
        ticktext=[]
    ))    
    fig.update_yaxes(dict(
        tickmode='array',
        tickvals=[],
        ticktext=[]
    ))


    # # Update axis limits:
    # fig.update_xaxes(range=[x_min - 2, x_min + span + 2])
    # fig.update_yaxes(range=[y_min - 2, y_min + span + 2])
    # Set aspect ratio to equal:
    fig.update_yaxes(
        scaleanchor='x',
        scaleratio=1,
    )

    # Remove y=0 and x=0 lines (zeroline) and grid lines:
    fig.update_yaxes(zeroline=False, showgrid=False)
    fig.update_xaxes(zeroline=False, showgrid=False)


    # Reduce size of figure by adjusting margins:
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=200)

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
