from .inputs import find_useful_dist_dict
from .added_utility_between_dists import \
    calculate_mean_changes, \
    find_weighted_change


def find_dist_dicts(time_to_ivt, time_to_mt):
    nlvo_ivt_dict = find_useful_dist_dict(
        'Non-large vessel (nLVO)', 'Intravenous thrombolysis (IVT)',
        time_to_ivt
        )
    lvo_ivt_dict = find_useful_dist_dict(
        'Large vessel (LVO)', 'Intravenous thrombolysis (IVT)',
        time_to_ivt
        )
    lvo_mt_dict = find_useful_dist_dict(
        'Large vessel (LVO)', 'Mechanical thrombectomy (MT)',
        time_to_mt
        )
    return nlvo_ivt_dict, lvo_ivt_dict, lvo_mt_dict


def find_outcome_dicts(
        nlvo_ivt_dict,
        lvo_ivt_dict,
        lvo_mt_dict,
        utility_weights,
        prop_dict
        ):

    mean_mRS_dict_nlvo_ivt, mean_util_dict_nlvo_ivt = \
        calculate_mean_changes(
            nlvo_ivt_dict['dist_pre_stroke'],
            nlvo_ivt_dict['dist_no_treatment'],
            nlvo_ivt_dict['dist_time_input_treatment'],
            utility_weights
            )
    mean_mRS_dict_lvo_ivt, mean_util_dict_lvo_ivt = \
        calculate_mean_changes(
            lvo_ivt_dict['dist_pre_stroke'],
            lvo_ivt_dict['dist_no_treatment'],
            lvo_ivt_dict['dist_time_input_treatment'],
            utility_weights
            )
    mean_mRS_dict_lvo_mt, mean_util_dict_lvo_mt = \
        calculate_mean_changes(
            lvo_mt_dict['dist_pre_stroke'],
            lvo_mt_dict['dist_no_treatment'],
            lvo_mt_dict['dist_time_input_treatment'],
            utility_weights
            )

    mean_mRS_change = find_weighted_change(
        mean_mRS_dict_lvo_ivt['diff_no_treatment'],
        mean_mRS_dict_lvo_mt['diff_no_treatment'],
        mean_mRS_dict_nlvo_ivt['diff_no_treatment'],
        prop_dict, util=False)
    mean_util_change = find_weighted_change(
        mean_util_dict_lvo_ivt['diff_no_treatment'],
        mean_util_dict_lvo_mt['diff_no_treatment'],
        mean_util_dict_nlvo_ivt['diff_no_treatment'],
        prop_dict)

    # Take mean population mRS from the input data:
    # (the pre-stroke and no-treatment dists are the same regardless
    # of treatment type. Only occlusion type matters.)
    mean_mRS_no_treatment = (
        prop_dict['lvo'] * mean_mRS_dict_lvo_mt['no_treatment'] +
        prop_dict['nlvo'] * mean_mRS_dict_nlvo_ivt['no_treatment']
    )
    mean_mRS_pre_stroke = (
        prop_dict['lvo'] * mean_mRS_dict_lvo_mt['pre_stroke'] +
        prop_dict['nlvo'] * mean_mRS_dict_nlvo_ivt['pre_stroke']
    )

    mean_util_no_treatment = (
        prop_dict['lvo'] * mean_util_dict_lvo_mt['no_treatment'] +
        prop_dict['nlvo'] * mean_util_dict_nlvo_ivt['no_treatment']
    )
    mean_util_pre_stroke = (
        prop_dict['lvo'] * mean_util_dict_lvo_mt['pre_stroke'] +
        prop_dict['nlvo'] * mean_util_dict_nlvo_ivt['pre_stroke']
    )

    mean_mRS_treated = mean_mRS_no_treatment + mean_mRS_change
    mean_util_treated = mean_util_no_treatment + mean_util_change

    mean_outcomes_dict_population = dict(
        mRS_no_treatment=mean_mRS_no_treatment,
        mRS_treated=mean_mRS_treated,
        mRS_change=mean_mRS_change,
        mRS_pre_stroke=mean_mRS_pre_stroke,
        util_no_treatment=mean_util_no_treatment,
        util_treated=mean_util_treated,
        util_change=mean_util_change,
        util_pre_stroke=mean_util_pre_stroke
    )

    return (
        mean_mRS_dict_nlvo_ivt,
        mean_util_dict_nlvo_ivt,
        mean_mRS_dict_lvo_ivt,
        mean_util_dict_lvo_ivt,
        mean_mRS_dict_lvo_mt,
        mean_util_dict_lvo_mt,
        mean_outcomes_dict_population
        )
