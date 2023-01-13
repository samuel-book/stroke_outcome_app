import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from outcome_utilities.fixed_params import emoji_text_dict

# Add an extra bit to the path if we need to.
# Try importing something as though we're running this from the same
# directory as the landing page.
try:
    emoji_image = plt.imread('./emoji/ambulance.png')
    dir = './'
except FileNotFoundError:
    # If the import fails, add the landing page directory to path.
    # Assume that the script is being run from the directory above
    # the landing page directory, which is called
    # stroke_outcome_app.
    dir = 'stroke_outcome_app/'


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

    fig = go.Figure()
    x_vals = [0, 1.2]
    label_offset = 0.6
    under_offset = -0.3
    emoji_offset = 0.1


    plotly_colours = px.colors.qualitative.Plotly

    y_max = 0.0

    for i, time_dict in enumerate(time_dicts):
        # plot_timeline_matplotlib(time_dict, ax, y=y_vals[i], emoji_dict=emoji_dict)
        x = x_vals[i]

        time_cumulative = 0.0
        time_cum_list = []
        time_cum_str_list = []
        time_colour_list = []
        write_under_list = []
        labels = []
        emoji_to_draw = []
        
        for time_key in time_dict.keys():
            # Store how to label this on the plot:
            labels.append(label_dict[time_key])
            try:
                emoji_to_draw.append(emoji_text_dict[time_key])
            except KeyError:
                emoji_to_draw.append('')
            # Find the cumulative time to this point:
            t_min = time_dict[time_key]
            time_cumulative += t_min/60.0
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
                write_under = True
            elif 'mt_arrival_to_treatment' in time_key:
                colour = plotly_colours[1]
                write_under = True
            else:
                # Default colour
                colour = None #'black'
                write_under = False
            time_colour_list.append(colour)
            write_under_list.append(write_under)

        # --- Plot ---
        # Make new labels list with line breaks removed:
        labels_plain = [l.replace('<br>', ' ') for l in labels]
        # Add straight line along the time axis:
        fig.add_trace(go.Scatter(
            y=time_cum_list,
            x=[x] * len(time_cum_list),
            mode='lines+markers',
            marker=dict(size=12, symbol='line-ew-open'),
            line=dict(color='grey'), #, width=lws[m]),
            showlegend=False,
            customdata=np.stack((time_cum_str_list, labels_plain), axis=-1)
        ))
        # 
        # Update the hover text for the lines:
        fig.update_traces(
            hovertemplate=(
                '%{customdata[1]}'
                '<br>' +
                'Time: %{customdata[0]}' +
                '<extra></extra>'
                )
            )

        # Add label for each scatter marker
        for t, time_cum in enumerate(time_cum_list):
            # Only show it if it's moved on from the previous:
            if t == 0 or time_cum_list[t] > time_cum_list[t-1]:
                if write_under_list[t] is True:
                    text = labels[t] + '<br>'+time_cum_str_list[t]
                else:
                    text = labels[t]
                fig.add_annotation(
                    y=time_cum,
                    x=x + label_offset,
                    text=text,
                    showarrow=False,
                    # yshift=1,
                    ax=0,  # Make arrow vertical - a = arrow, x = x-shift.
                    font=dict(color=time_colour_list[t]),
                    )
                # Add emoji for each scatter marker
                fig.add_annotation(
                    y=time_cum,
                    x=x + emoji_offset,
                    text=emoji_to_draw[t],
                    showarrow=False,
                    # yshift=1,
                    ax=0,  # Make arrow vertical - a = arrow, x = x-shift.
                    font=dict(color=time_colour_list[t])
                    )
                # if write_under_list[t] is True:
                #     fig.add_annotation(
                #         y=time_cum,
                #         x= x + under_offset,
                #         text=time_cum_str_list[t],
                #         showarrow=False,
                #         yshift=1,
                #         ax=0,  # Make arrow vertical - a = arrow, x = x-shift.
                #         font=dict(color=time_colour_list[t]),
                #         textangle=-45
                #         )

            if time_cumulative > y_max:
                y_max = time_cumulative


    # Set y range:
    fig.update_yaxes(range=[y_max * 1.05, 0 - y_max * 0.025]) #y_max])
    # (don't need the following when manual axis limits are set:)
    # Flip y-axis so values are displayed as positive but 0 is on top.
    # fig.update_layout(yaxis=dict(autorange='reversed'))

    # Set x-axis labels
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=x_vals,
            ticktext=['<b>Case 1', '<b>Case 2'],
            # tickfont=dict(weight='heavy'),#)color='darkgray'),
            side='top'  # Moves the labels to the top of the grid
        ),
    )
    # Set y-axis label
    fig.update_yaxes(title_text='Time since onset (hours)')

    # for i, time_dict in enumerate(time_dicts):
    #     plot_emoji_on_timeline(ax, emoji_dict, time_dict, y=y_vals[i],
    #                            xlim=xlim, ylim=ylim)


    # Remove y=0 and x=0 lines (zeroline) and grid lines:
    fig.update_yaxes(zeroline=False, showgrid=False)
    fig.update_xaxes(zeroline=False, showgrid=False)

    # Reduce size of figure by adjusting margins:
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=500)

    # Write to streamlit:
    st.plotly_chart(fig, use_container_width=True)


def plot_timeline_matplotlib(time_dict, ax=None, y=0, emoji_dict={}):
    label_dict = dict(
        onset='Onset',
        onset_to_ambulance_arrival='Travel to\nhospital\nbegins',
        travel_to_ivt='Arrive at\nIVT\ncentre',
        travel_to_mt='Arrive at\nIVT+MT\ncentre',
        ivt_arrival_to_treatment='IVT',
        transfer_additional_delay='Transfer\nbegins',
        travel_ivt_to_mt='Arrive at\nIVT+MT\ncentre',
        mt_arrival_to_treatment='MT',
        )

    if ax is None:
        fig, ax = plt.subplots()
    time_cumulative = 0.0
    y_under_offset = -0.05
    y_label_offset = 0.30
    for time_key in time_dict.keys():
        t_min = time_dict[time_key]
        time_cumulative += t_min/60.0

        if 'ivt_arrival_to_treatment' in time_key:
            colour = 'b'
            write_under = True
        elif 'mt_arrival_to_treatment' in time_key:
            colour = 'r'
            write_under = True
        else:
            colour = 'k'
            write_under = False

        if time_dict[time_key] == 0.0 and time_key != 'onset':
            x_plot = np.NaN
        else:
            x_plot = time_cumulative
        ax.scatter(x_plot, y, marker='|', s=200, color=colour)

        ax.annotate(
            label_dict[time_key], xy=(x_plot, y+y_label_offset),
            rotation=0, color=colour, ha='center', va='bottom')
        if write_under:
            time_str = (f'{int(60*time_cumulative//60):2d}hr ' +
                        f'{int(60*time_cumulative%60):2d}min')
            ax.annotate(
                time_str, xy=(x_plot, y+y_under_offset), color=colour,
                ha='center', va='top', rotation=20)
    ax.plot([0, time_cumulative], [y, y], color='k', zorder=0)


def plot_emoji_on_timeline(
        ax, emoji_dict, time_dict, y=0,
        xlim=[], ylim=[]
        ):
    """Do this after the timeline is drawn so sizing is consistent."""

    y_emoji_offset = 0.15

    y_span = ylim[1] - ylim[0]
    y_size = 1.5*0.07*y_span

    x_span = xlim[1] - xlim[0]
    x_size = 1.5*0.04*x_span

    time_cumulative = 0.0
    for time_key in time_dict.keys():
        if time_key in emoji_dict.keys():
            t_min = time_dict[time_key]
            time_cumulative += t_min/60.0
            if time_dict[time_key] == 0.0 and time_key != 'onset':
                x_plot = np.NaN
            else:
                x_plot = time_cumulative

                emoji = emoji_dict[time_key].strip(':')
                # Import from file
                emoji_image = plt.imread(dir + 'emoji/' + emoji + '.png')
                ext_xmin = x_plot - x_size*0.5
                ext_xmax = x_plot + x_size*0.5
                ext_ymin = y+y_emoji_offset - y_size*0.5
                ext_ymax = y+y_emoji_offset + y_size*0.5

                if 'ambulance' not in emoji:
                    extent = [ext_xmin, ext_xmax, ext_ymin, ext_ymax]
                else:
                    # If it's the ambulance emoji, flip it horizontally:
                    extent = [ext_xmax, ext_xmin, ext_ymin, ext_ymax]

                ax.imshow(emoji_image, extent=extent)


def make_timeline_plot_matplotlib(ax, time_dicts, emoji_dict={}):

    y_step = 1.0
    y_vals = np.arange(0.0, y_step*len(time_dicts), y_step)[::-1]
    for i, time_dict in enumerate(time_dicts):
        plot_timeline_matplotlib(time_dict, ax, y=y_vals[i], emoji_dict=emoji_dict)

    xlim = ax.get_xlim()
    ax.set_xticks(np.arange(0, xlim[1], (1+(xlim[1]//6))*(10/60)), minor=True)
    ax.set_xlabel('Time since onset (hours)')

    ax.set_ylim(-0.25, y_step*(len(time_dicts)-0.2))
    ylim = ax.get_ylim()
    ax.set_yticks(y_vals)
    ax.set_yticklabels(
        [f'Case {i+1}' for i in range(len(time_dicts))], fontsize=14)

    aspect = 1.0/(ax.get_data_ratio()*2)
    ax.set_aspect(aspect)
    for i, time_dict in enumerate(time_dicts):
        plot_emoji_on_timeline(ax, emoji_dict, time_dict, y=y_vals[i],
                               xlim=xlim, ylim=ylim)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect(aspect)

    for spine in ['top', 'left', 'right']:
        ax.spines[spine].set_color('w')
