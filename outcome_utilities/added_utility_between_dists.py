"""
Function to compare two mRS probability distributions and find the
difference in utility across the full distributions.

e.g. when both probability distributions have mRS=0, the difference
in utility is 0. When one has mRS=0 and the other mRS=1, the difference
is 0.97 - 0.88 = 0.09.
"""
import numpy as np


def find_added_utility_between_dists(
        mRS_dist1,
        mRS_dist2,
        utility_weights=[]
        ):
    """
    Find the difference in utility between two mRS probability
    distributions.

    Inputs:
    mRS_dist1       - np.array. mRS probability distribution 1.
    mRS_dist2       - np.array. mRS probability distribution 2.
    utility_weights - list or np.array. The utility weighting given to
                      each mRS value. If none given, a default is used.

    Returns:
    mRS_dist_mix - np.array. Cumulative probability distribution of
                   mRS_dist1 and mRS_dist2 combined and sorted.
    added_utils  -
    mRS_diff_mix - np.array. Non-cumulative probability distribution of
                   mRS_dist1 and mRS_dist2 combined and sorted.
    """
    if len(utility_weights) < 1:
        utility_weights = np.array(
            [0.97, 0.88, 0.74, 0.55, 0.20, -0.19, 0.00])

    # Combine the two mRS distributions into one ordered list:
    mRS_dist_mix = np.concatenate((mRS_dist1, mRS_dist2))
    # Sort and remove the duplicate 1.0 at the end:
    mRS_dist_mix = np.sort(mRS_dist_mix)[:-1]
    # Add a 0.0 at the start:
    mRS_dist_mix = np.concatenate(([0.0], mRS_dist_mix))

    # Find the size of each bin (not cumulative):
    mRS_diff_mix = np.diff(mRS_dist_mix, prepend=0.0)

    # Store the mRS indices in here:
    x1_list = []
    x2_list = []
    # And store the utility values in here:
    u1_list = []
    u2_list = []
    for i, boundary in enumerate(mRS_dist_mix):
        # Find which mRS bin we're currently in:
        x1 = np.digitize(boundary, mRS_dist1, right=True)
        x2 = np.digitize(boundary, mRS_dist2, right=True)

        # Store values:
        x1_list.append(x1)
        x2_list.append(x2)
        u1_list.append(utility_weights[x1])
        u2_list.append(utility_weights[x2])

    # Find the increase in utility between dists 1 and 2:
    added_utils = np.array(u1_list) - np.array(u2_list)

    # Weight the increases by the proportion of the mRS distribution
    # that they span:
    weighted_added_utils = np.cumsum(added_utils * mRS_diff_mix)

    # Round the distribution values to three decimal places
    # - might not add up to 1 afterwards, but saves apparent rounding
    # errors in the printed utility and mRS change sums.
    mRS_dist_mix = np.round(mRS_dist_mix, 3)

    return mRS_dist_mix, weighted_added_utils, x1_list, x2_list


def calculate_mean_changes(
        dist_pre_stroke, dist_no_treatment,
        dist_time_input_treatment, utility_weights
        ):
    # ----- Calculate metrics -----
    # Calculate mean mRSes:
    mean_mRS_pre_stroke = np.sum(
        dist_pre_stroke*np.arange(7))
    mean_mRS_no_treatment = np.sum(dist_no_treatment*np.arange(7))
    mean_mRS_time_input_treatment = np.sum(
        dist_time_input_treatment*np.arange(7))
    # Differences:
    mean_mRS_diff_no_treatment = (
        mean_mRS_time_input_treatment - mean_mRS_no_treatment)
    mean_mRS_diff_pre_stroke = (
        mean_mRS_time_input_treatment - mean_mRS_pre_stroke)

    # Gather:
    mean_mRS_dict = dict(
        pre_stroke=mean_mRS_pre_stroke,
        no_treatment=mean_mRS_no_treatment,
        time_input_treatment=mean_mRS_time_input_treatment,
        diff_no_treatment=mean_mRS_diff_no_treatment,
        diff_pre_stroke=mean_mRS_diff_pre_stroke
    )

    # Calculate mean utilities:
    # (it seems weird to use "sum" instead of "mean" but this definition
    # matches the clinical outcome script)
    mean_utility_pre_stroke = np.sum(
        dist_pre_stroke*utility_weights)
    mean_utility_no_treatment = np.sum(dist_no_treatment*utility_weights)
    mean_utility_time_input_treatment = np.sum(
        dist_time_input_treatment*utility_weights)
    # Differences:
    mean_utility_diff_no_treatment = (
        mean_utility_time_input_treatment - mean_utility_no_treatment)
    mean_utility_diff_pre_stroke = (
        mean_utility_time_input_treatment - mean_utility_pre_stroke)

    # Gather:
    mean_util_dict = dict(
        pre_stroke=mean_utility_pre_stroke,
        no_treatment=mean_utility_no_treatment,
        time_input_treatment=mean_utility_time_input_treatment,
        diff_no_treatment=mean_utility_diff_no_treatment,
        diff_pre_stroke=mean_utility_diff_pre_stroke
    )

    return mean_mRS_dict, mean_util_dict


def calculate_combo_mean_changes(
        prop_dict, change_dict_nlvo_ivt, change_dict_lvo_ivt,
        change_dict_lvo_mt, change_key):
    p1 = prop_dict['lvo']*prop_dict['lvo_treated_ivt_only']
    p2 = prop_dict['lvo']*prop_dict['lvo_treated_ivt_mt']
    p3 = prop_dict['nlvo']*prop_dict['nlvo_treated_ivt_only']
    mean_change = (
        p1*change_dict_lvo_ivt[change_key] +
        p2*change_dict_lvo_mt[change_key] +
        p3*change_dict_nlvo_ivt[change_key]
    )
    return mean_change


def find_weighted_change(change_lvo_ivt, change_lvo_mt, change_nlvo_ivt,
                         patient_props, util=True):
    """
    Take the total changes for each category and calculate their
    weighted sum, where weights are from the proportions of the
    patient population.

    (originally from matrix notebook)

    Inputs:

    Returns:

    """
    # If LVO-IVT is greater change than LVO-MT then adjust MT for
    # proportion of patients receiving IVT:
    adjust = False
    if util and change_lvo_ivt > change_lvo_mt:
        adjust = True
    elif util is False and change_lvo_ivt < change_lvo_mt:
        adjust = True
    if adjust:
        diff = change_lvo_ivt - change_lvo_mt
        change_lvo_mt += diff * patient_props['lvo_mt_also_receiving_ivt']

    # Calculate weighted changes (wc):
    wc_lvo_mt = (
        change_lvo_mt *
        patient_props['lvo'] *
        patient_props['lvo_treated_ivt_mt']
        )
    wc_lvo_ivt = (
        change_lvo_ivt *
        patient_props['lvo'] *
        patient_props['lvo_treated_ivt_only']
        )
    wc_nlvo_ivt = (
        change_nlvo_ivt *
        patient_props['nlvo'] *
        patient_props['nlvo_treated_ivt_only']
        )

    total_change = wc_lvo_mt + wc_lvo_ivt + wc_nlvo_ivt
    return total_change
