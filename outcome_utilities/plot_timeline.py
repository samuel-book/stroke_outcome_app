import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import plotly.graph_objects as go


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


def make_timeline_plot(time_dicts, emoji_dict={}):
    label_dict = dict(
        onset='Onset',
        onset_to_ambulance_arrival='Travel to<br>hospital<br>begins',
        travel_to_ivt='Arrive at<br>IVT<br>centre',
        travel_to_mt='Arrive at<br>IVT+MT<br>centre',
        ivt_arrival_to_treatment='IVT',
        transfer_additional_delay='Transfer<br>begins',
        travel_ivt_to_mt='Arrive at<br>IVT+MT<br>centre',
        mt_arrival_to_treatment='MT',
        )

    fig = go.Figure()
    x_vals = [0, 1]
    label_offset = 0.3
    under_offset = -0.3

    for i, time_dict in enumerate(time_dicts):
        # plot_timeline_matplotlib(time_dict, ax, y=y_vals[i], emoji_dict=emoji_dict)
        x = x_vals[i]

        time_cumulative = 0.0
        time_cum_list = []
        time_cum_str_list = []
        time_colour_list = []
        write_under_list = []
        labels = []
        
        for time_key in time_dict.keys():
            # Store how to label this on the plot:
            labels.append(label_dict[time_key])
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
                colour = 'blue'
                write_under = True
            elif 'mt_arrival_to_treatment' in time_key:
                colour = 'red'
                write_under = True
            else:
                colour = 'black'
                write_under = False
            time_colour_list.append(colour)
            write_under_list.append(write_under)

        # --- Plot ---
        # Add straight line along the time axis:
        fig.add_trace(go.Scatter(
            y=time_cum_list,
            x=[x] * len(time_cum_list),
            mode='lines+markers',
            marker=dict(size=20, symbol='line-ew-open'),
            line=dict(color='black'),
            showlegend=False,
            customdata=np.stack((time_cum_str_list, labels), axis=-1)
        ))
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
            fig.add_annotation(
                y=time_cum,
                x=x + label_offset,
                text=labels[t],
                showarrow=False,
                # yshift=1,
                ax=0,  # Make arrow vertical - a = arrow, x = x-shift.
                font=dict(color=time_colour_list[t])
                )
            if write_under_list[t] is True:
                fig.add_annotation(
                    y=time_cum,
                    x= x + under_offset,
                    text=time_cum_str_list[t],
                    showarrow=False,
                    yshift=1,
                    ax=0,  # Make arrow vertical - a = arrow, x = x-shift.
                    font=dict(color=time_colour_list[t])
                    )

    # Flip y-axis so values are displayed as positive but 0 is on top.
    fig['layout']['yaxis']['autorange'] = 'reversed'

    # Set x-axis labels
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=x_vals,
            ticktext=['Case 1', 'Case 2'],
            # tickfont=dict(color='darkgray'),
            side='top'  # Moves the labels to the top of the grid
        ),
    )
    # Set y-axis label
    fig.update_yaxes(title_text='Time since onset (hours)')

    # for i, time_dict in enumerate(time_dicts):
    #     plot_emoji_on_timeline(ax, emoji_dict, time_dict, y=y_vals[i],
    #                            xlim=xlim, ylim=ylim)


    # Reduce size of figure by adjusting margins:
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=800)

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
