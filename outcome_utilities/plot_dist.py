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
