"""
Now-unused matplotlib versions of plots.


# plot_timeline_matplotlib(time_dict, ax, y=y_vals[i], emoji_dict=emoji_dict)

# for i, time_dict in enumerate(time_dicts):
#     plot_emoji_on_timeline(ax, emoji_dict, time_dict, y=y_vals[i],
#                            xlim=xlim, ylim=ylim)

"""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


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
        plot_timeline_matplotlib(
            time_dict, ax, y=y_vals[i], emoji_dict=emoji_dict
            )

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



def make_fig_legend(colour_list):
    """
    Plot a legend for the mRS colours.
    """
    fig_legend = plt.figure(figsize=(6, 2))
    # Dummy data for legend:
    dummies = []
    for i in range(7):
        dummy = plt.bar(np.NaN, np.NaN, color=colour_list[i], edgecolor='k')
        dummies.append(dummy)

    # Clear to remove automatic blank axis:
    fig_legend.clear()
    # Draw legend using dummy bars:
    fig_legend.legend(
        [*dummies], range(7),
        loc='center', ncol=7, title='mRS colour scheme',
        )
    return fig_legend


def do_probs_with_time_matplotlib(
        time_no_effect, A_list, b_list, colour_list, treatment_times,
        treatment_labels=[], time_no_effect_mt=8*60
        ):
    # --- Plot probability with time -----
    times_to_plot = np.linspace(0, time_no_effect, 20)
    # figsize is fudged to make it approx. same height as other plot
    fig_probs_time, ax_probs_time = plt.subplots(figsize=(8, 4))
    plot_probs_filled(
        A_list, b_list, times_to_plot, colour_list,
        # probs_to_mark=np.unique(probs_to_mark),
        treatment_times, treatment_labels,
        ax=ax_probs_time, xmax=time_no_effect_mt/60)
    st.pyplot(fig_probs_time)


def plot_probs_filled(
        A,
        b,
        times_mins,
        colour_list=[],
        treatment_times=[],
        treatment_labels=[],
        ax=None,
        title='',
        xmax=None
        ):
    if ax is None:
        ax = plt.subplot()
    # P(mRS<=5)=1.0 at all times, so it has no defined A, a, and b.
    # Instead append to this array a 0.0, which won't be used directly
    # but will allow the "for" loop to go round one extra time.
    A = np.append(A, 0.0)
    times_hours = times_mins/60.0

    p_j = np.zeros(times_mins.shape)
    for i, A_i in enumerate(A):
        if len(colour_list) > 0:
            colour = colour_list[i]
        else:
            colour = None
        # Define the probability line, p_i:
        if i < 6:
            p_i = 1.0/(1.0 + np.exp(-A_i - b[i]*times_mins))
        else:
            # P(mRS<=5)=1.0 at all times:
            p_i = np.full(times_mins.shape, 1.0)

        # Plot it as before and store the colour used:
        ax.plot(times_hours, p_i, color='k', linewidth=1)
        # Fill the area between p_i and the line below, p_j.
        # This area marks where mRS <= the given value.
        # If p_j is not defined yet (i=0), set all p_j to zero:
        p_j = p_j if i > 0 else np.zeros_like(p_i)
        ax.fill_between(times_hours, p_i, p_j, label=f'{i}',
                        color=colour)

        # Store the most recently-created line for the next loop:
        p_j = p_i

    if len(treatment_times) > 0:
        if len(treatment_labels) != len(treatment_times):
            treatment_labels = ['Treatment' for t in treatment_times]
        for i, time_input in enumerate(treatment_times):
            ax.axvline(time_input/60.0, color='k', linestyle=':')
            ax.annotate(
                '|',
                xy=(time_input/60.0, 0.95),
                va='bottom', ha='center', color='r',
                fontsize=20, zorder=0
                )
            ax.annotate(
                treatment_labels[i]+'\n',
                xy=(time_input/60.0, 1.0),
                va='bottom', ha='left', color='r',
                rotation=30
                )

    ax.set_ylabel('Probability')
    ax.set_xlabel('Onset to treatment time (hours)')

    ax.set_ylim(0, 1)

    if xmax is None:
        xmax = times_hours[-1]

    # Secret annotation to make sure axis doesn't resize when time input
    # is equal to max time:
    ax.annotate('Treatment',
                xy=(xmax, 0.0), va='top', ha='center', color='None')

    ax.set_xlim(times_hours[0], xmax)

    ax.set_xticks(np.arange(times_hours[0], xmax+0.01, 1))
    ax.set_xticks(np.arange(times_hours[0], xmax+0.01, 0.25), minor=True)

    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.05), minor=True)
    ax.tick_params(top=True, bottom=True, left=True, right=True, which='both')

    ax.set_title(title)

"""
Functions for plotting mRS distributions as horizontal bars.
"""
import matplotlib.pyplot as plt


def draw_horizontal_bar(dist,
                        y=0, colour_list=[], hatch_list=[],
                        ecolour_list=[], linewidth=None, bar_height=0.5,
                        ax=None):
    """
    Draw a stacked horizontal bar chart of the values in 'dist'.

    dist  - list or np.array. The probability distribution
            (non-cumulative).
    label - string. The name printed next to these stacked bars.
    """
    # Define any missing inputs:
    if ax is None:
        ax = plt.subplot()
    if len(colour_list) < 1:
        # Get current matplotlib style sheet colours:
        colour_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
        while len(colour_list) < len(dist):
            # Shouldn't need this, but in case dist is really long:
            colour_list += colour_list
        # Remove extra colours:
        colour_list = colour_list[:len(dist)-1]
        # Add grey for mRS=6 bin:
        colour_list.append('DarkSlateGray')
    if len(hatch_list) < 1:
        hatch_list = [None for d in dist]
    if len(ecolour_list) < 1:
        ecolour_list = ['k' for d in dist]

    # The first bar will start at this point on the x-axis:
    left = 0
    for i in range(len(dist)):
        # Don't add bar to legend if it has fancy hatch formatting:
        # legend_label = f'{i%7}' if hatch_list[i]==None else None
        legend_label = (
            None
            if hatch_list[i] is not None and '\\' in hatch_list[i]
            else f'{i%7}'
            )

        # Draw a bar starting from 'left', the end of the previous bar,
        # with a width equal to the probability of this mRS:
        ax.barh(
            y,
            width=dist[i],
            left=left,
            height=bar_height,
            label=legend_label,
            edgecolor=ecolour_list[i],
            color=colour_list[i],
            hatch=hatch_list[i],
            linewidth=linewidth,
            # tick_label=label
        )
        # Update 'left' with the width of the current bar so that the
        # next bar drawn will start in the correct place.
        left += dist[i]


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
def draw_connections(dist_top, dist_bottom,
                     top_of_bottom_bar=0.25, bottom_of_top_bar=0.75,
                     colour='k', linewidth=1.0, ax=None):
    """
    Draw lines connecting the mRS bins in the top and bottom rows.

    dist_t0, dist_tne - lists or arrays. Probability distributions.
    top_tne, bottom_t0 - floats. y-coordinates just inside the bars.
    """

    # Define any missing inputs:
    if ax is None:
        ax = plt.subplot()

    left_of_top_bar = 0.0
    left_of_bottom_bar = 0.0
    for i, d_t0 in enumerate(dist_top):
        left_of_top_bar += dist_top[i]
        left_of_bottom_bar += dist_bottom[i]
        ax.plot([left_of_top_bar, left_of_bottom_bar],
                [bottom_of_top_bar, top_of_bottom_bar],
                color=colour, linewidth=linewidth)


def do_prob_bars_matplotlib(
        dist_dict,
        mRS_dist_mix, weighted_added_utils,
        mRS_list_time_input_treatment,
        mRS_list_no_treatment, time_input, key_str=''
        ):
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
        ax_bars, time_input, y_list, bar_height
        )

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

