import numpy as np


def find_bin_size_ratios(
        dist,
        dist_cumsum,
        y_bar,
        pre_stroke_bin_left,
        pre_stroke_bin_right,
        pre_stroke_bin_size
        ):
    # , bar_height=0.5):

    size_ratios = []
    bin_mids = []

    # Find which mRS bins here are within the highlight:
    bin_smallest = np.where(dist_cumsum >= pre_stroke_bin_left)[0][0]
    bin_largest = np.where(dist_cumsum >= pre_stroke_bin_right)[0][0]

    for mRS_bin in range(bin_smallest, bin_largest+1, 1):
        bin_size = dist[mRS_bin]
        if mRS_bin == bin_smallest:
            bin_size -= pre_stroke_bin_left
            bin_left = pre_stroke_bin_left
        else:
            bin_left = dist_cumsum[mRS_bin-1] if mRS_bin > 0 else 0.0
        if mRS_bin == bin_largest:
            bin_size = pre_stroke_bin_right-bin_size
            bin_right = pre_stroke_bin_right
        else:
            bin_right = dist_cumsum[mRS_bin]
        # Central coordinate:
        bin_mid = np.mean([bin_left, bin_right])

        # Find size ratio of this bin to the highlight:
        size_ratio = (bin_right-bin_left)/pre_stroke_bin_size

        # Store in list:
        size_ratios.append(size_ratio)
        bin_mids.append(bin_mid)

        # # Annotate the size label:
        # text = ax_bars.annotate(f'{size_ratio*100:2.0f}%',
        #     xy=(bin_mid, y_bar+bar_height*0.6),
        #     ha='center', va='bottom', color=colour_list[mRS],
        #     fontsize=8)
        # # Add white outline:
        # text.set_path_effects([
        #     path_effects.Stroke(linewidth=3, foreground='w'),
        #     path_effects.Normal()])

    return size_ratios, bin_mids
