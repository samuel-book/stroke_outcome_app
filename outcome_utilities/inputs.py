import streamlit as st
import numpy as np
import pandas as pd

from .fixed_params import emoji_text_dict, utility_weights
from .probs_with_time import find_dists_at_chosen_time
from .added_utility_between_dists import \
    find_added_utility_between_dists


# Add an extra bit to the path if we need to.
# Try importing something as though we're running this from the same
# directory as the landing page.
try:
    test_file = pd.read_csv(
        './outcome_data/mrs_dist_probs_cumsum.csv',
        index_col='Stroke type'
        )
    dir = './'
except FileNotFoundError:
    # If the import fails, add the landing page directory to path.
    # Assume that the script is being run from the directory above
    # the landing page directory, which is called
    # stroke_outcome_app.
    dir = 'stroke_outcome_app/'


def write_text_from_file(filename, head_lines_to_skip=0):
    """
    Write text from 'filename' into streamlit.
    Skip a few lines at the top of the file using head_lines_to_skip.
    """
    # Open the file and read in the contents,
    # skipping a few lines at the top if required.
    try:
        with open(filename, 'r', encoding="utf-8") as f:
            text_to_print = f.readlines()[head_lines_to_skip:]
    except FileNotFoundError:
        with open('stroke_outcome_app/' + filename, 'r', encoding="utf-8") as f:
            text_to_print = f.readlines()[head_lines_to_skip:]

    # Turn the list of all of the lines into one long string
    # by joining them up with an empty '' string in between each pair.
    text_to_print = ''.join(text_to_print)

    # Write the text in streamlit.
    st.markdown(f"""{text_to_print}""")


def inputs_pathway(pathway_cols):
    # All fixed times have units of minutes
    # pathway_cols = st.columns(3)

    # Column 1
    with pathway_cols[0]:
        st.markdown('__Both cases:__')
        # Case 1 + 2
        onset_to_ambulance_arrival = st.number_input(
            label=(emoji_text_dict['onset_to_ambulance_arrival'] +
                   ' Onset to ambulance arrival'),
            min_value=0,
            max_value=600,
            value=60,
            step=5
            )
        # Case 1 + 2
        ivt_arrival_to_treatment = st.number_input(
            label=(
                emoji_text_dict['ivt_arrival_to_treatment'] +
                ' Delay between arrival at either centre and IVT treatment'
                ),
            min_value=0,
            max_value=240,
            value=30,
            step=5
            )
        # Case 1 + 2
        mt_arrival_to_treatment = st.number_input(
            label=(
                emoji_text_dict['mt_arrival_to_treatment'] +
                ' Delay between arrival at IVT+MT centre and MT treatment'
                ),
            min_value=0,
            max_value=240,
            value=90,
            step=5
            )


    # Column 2
    with pathway_cols[1]:
        st.write('__Case 1 only:__')
        # Case 1 only
        travel_to_ivt = st.number_input(
            label=(emoji_text_dict['travel_to_ivt'] +
                   ' Travel time from onset location to IVT centre'),
            min_value=0,
            max_value=120,
            value=30,
            step=5
            )
        # Case 1 only
        transfer_additional_delay = st.number_input(
            label=(emoji_text_dict['onset_to_ambulance_arrival'] +
                   ' Delay for transfer between centres'),
            min_value=0,
            max_value=180,
            value=60,
            step=5
            )
        # Case 1 only
        travel_ivt_to_mt = st.number_input(
            label=(emoji_text_dict['travel_ivt_to_mt'] +
                   ' Travel time between IVT and IVT+MT centres'),
            min_value=0,
            max_value=180,
            value=50,
            step=5
            )
        st.write('__Case 2 only:__')
        # Case 2 only
        travel_to_mt = st.number_input(
            label=(emoji_text_dict['travel_to_mt'] +
                   ' Travel time from onset location to IVT+MT centre'),
            min_value=0,
            max_value=120,
            value=50,
            step=5
            )


    # ----- end of inputs -----

    case1_time_dict = dict(
        onset=0,
        onset_to_ambulance_arrival=onset_to_ambulance_arrival,
        travel_to_ivt=travel_to_ivt,
        ivt_arrival_to_treatment=ivt_arrival_to_treatment,
        transfer_additional_delay=transfer_additional_delay,
        travel_ivt_to_mt=travel_ivt_to_mt,
        mt_arrival_to_treatment=mt_arrival_to_treatment,
    )

    case2_time_dict = dict(
        onset=0,
        onset_to_ambulance_arrival=onset_to_ambulance_arrival,
        travel_to_mt=travel_to_mt,
        ivt_arrival_to_treatment=ivt_arrival_to_treatment,
        mt_arrival_to_treatment=np.max([
            mt_arrival_to_treatment-ivt_arrival_to_treatment, 1]),
        # Always at least one minute after IVT before MT
    )

    # Calculate times to treatment:
    case1_time_to_ivt = np.sum([
        onset_to_ambulance_arrival,
        travel_to_ivt,
        ivt_arrival_to_treatment
    ])

    case1_time_to_mt = case1_time_to_ivt + np.sum([
        transfer_additional_delay,
        travel_ivt_to_mt,
        mt_arrival_to_treatment
    ])

    case2_time_to_ivt = np.sum([
        onset_to_ambulance_arrival,
        travel_to_mt,
        ivt_arrival_to_treatment
    ])

    case2_time_to_mt = case2_time_to_ivt + np.sum([
        -ivt_arrival_to_treatment,
        mt_arrival_to_treatment
    ])

    return (case1_time_dict, case2_time_dict, case1_time_to_ivt,
            case1_time_to_mt, case2_time_to_ivt, case2_time_to_mt)


def inputs_simple_times(pathway_cols):
    # All fixed times have units of minutes
    # pathway_cols = st.columns(3)

    # Column 1
    with pathway_cols[0]:
        st.markdown('__Case 1:__')
        case1_ivt_input = st.number_input(
            label='Time to IVT',
            min_value=0,
            max_value=600,
            value=120,
            step=5,
            key='Case1_IVT'
            )
        case1_mt_input = st.number_input(
            label='Time to MT',
            min_value=0,
            max_value=600,
            value=320,
            step=5,
            key='Case1_MT'
            )


    # Column 1
    with pathway_cols[1]:
        st.markdown('__Case 2:__')
        case2_ivt_input = st.number_input(
            label='Time to IVT',
            min_value=0,
            max_value=600,
            value=140,
            step=5,
            key='Case2_IVT'
            )
        case2_mt_input = st.number_input(
            label='Time to MT',
            min_value=0,
            max_value=600,
            value=200,
            step=5,
            key='Case2_MT'
            )

    # ----- end of inputs -----

    case1_time_to_ivt = case1_ivt_input
    case1_time_to_mt = case1_mt_input
    case2_time_to_ivt = case2_ivt_input
    case2_time_to_mt = case2_mt_input

    case1_time_dict = dict(
        onset=0,
        time_to_ivt=case1_time_to_ivt,
        time_to_mt=case1_time_to_mt
    )

    case2_time_dict = dict(
        onset=0,
        time_to_ivt=case2_time_to_ivt,
        time_to_mt=case2_time_to_mt
    )

    return (case1_time_dict, case2_time_dict, case1_time_to_ivt,
            case1_time_to_mt, case2_time_to_ivt, case2_time_to_mt)


def inputs_patient_population():
    with st.form(key='form_props'):
        st.write('Percentage of patients with each stroke type:')
        # form_cols = st.columns(3)
        prop_nlvo = st.number_input(
            label='nLVO',
            min_value=0,
            max_value=100,
            value=65
            )
        prop_lvo = st.number_input(
            label='LVO',
            min_value=0,
            max_value=100,
            value=35
            )
        # prop_ich = st.number_input(
        #     label='ICH',
        #     min_value=0,
        #     max_value=100,
        #     value=0
        #     )

        # Convert from percentage to fraction:
        prop_nlvo *= 0.01
        prop_lvo *= 0.01
        # prop_ich *= 0.01

        # Sanity check - do the proportions sum to 100%?
        sum_props = np.sum([prop_nlvo, prop_lvo])
        if sum_props != 1:
            st.warning(':warning: Proportions should sum to 100%.')

        st.form_submit_button(label='Submit')

    prop_dict = dict(
        nlvo=prop_nlvo,
        lvo=prop_lvo,
        # ich=prop_ich
    )
    return prop_dict


def inputs_patient_population_advanced(prop_dict):
    # ----- Advanced options for patient proportions -----
    st.write('Percentage of each stroke type given each treatment: ')
    with st.form(key='form_props_treatment'):
        # for i,col in enumerate(form_cols):
        #     col.write('-'*20)
        # form_cols = st.columns(3)
        # form_cols = st.columns(5)
        prop_nlvo_treated_ivt_only = st.number_input(
            label=(emoji_text_dict['ivt_arrival_to_treatment'] +
                    ' nLVO given IVT'),
            min_value=0.0,
            max_value=100.0,
            value=15.5,
            step=0.1,
            format='%3.1f'
            )
        prop_lvo_treated_ivt_only = st.number_input(
            label=(emoji_text_dict['ivt_arrival_to_treatment'] +
                    ' LVO given IVT only'),
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.1,
            format='%3.1f'
            )
        prop_lvo_treated_ivt_mt = st.number_input(
            label=(emoji_text_dict['mt_arrival_to_treatment'] +
                    ' LVO given MT'),
            min_value=0.0,
            max_value=100.0,
            value=28.6,
            step=0.1,
            format='%3.1f'
            )
        # prop_ich_treated = st.number_input(
        #     label='ICH treated',
        #     min_value=0.0,
        #     max_value=100.0,
        #     value=0.0,
        #     step=0.1,
        #     format='%3.1f'
        #     )

        prop_lvo_mt_also_receiving_ivt = st.number_input(
            label=(
                emoji_text_dict['ivt_arrival_to_treatment'] +
                emoji_text_dict['mt_arrival_to_treatment'] +
                ' LVO MT patients who also receive IVT'),
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=0.1,
            format='%3.1f'
            )

        # Convert from percentage to fraction:
        prop_nlvo_treated_ivt_only *= 0.01
        prop_lvo_treated_ivt_only *= 0.01
        prop_lvo_treated_ivt_mt *= 0.01
        # prop_ich_treated *= 0.01
        prop_lvo_mt_also_receiving_ivt *= 0.01

        st.form_submit_button(label='Submit')

    treated_population = (
        prop_dict['nlvo'] * prop_nlvo_treated_ivt_only +
        prop_dict['lvo'] * prop_lvo_treated_ivt_mt +
        prop_dict['lvo'] * prop_lvo_treated_ivt_only
        )

    # Update the dictionary of patient proportion information:
    prop_dict['nlvo_treated_ivt_only'] = prop_nlvo_treated_ivt_only
    prop_dict['lvo_treated_ivt_only'] = prop_lvo_treated_ivt_only
    prop_dict['lvo_treated_ivt_mt'] = prop_lvo_treated_ivt_mt
    # prop_dict['ich_treated'] = prop_ich_treated
    prop_dict['lvo_mt_also_receiving_ivt'] = prop_lvo_mt_also_receiving_ivt
    prop_dict['treated_population'] = treated_population

    return prop_dict


def find_mRS_dists_from_file(occlusion_str, treatment_str):
    """
    Import the mRS data from file.

    If no data exists, return warning string.
    """
    # Load mRS distributions from file:
    mrs_dists_cumsum = pd.read_csv(
        dir + 'outcome_data/mrs_dist_probs_cumsum.csv',
        index_col='Stroke type')
    mrs_dists_bins = pd.read_csv(
        dir + 'outcome_data/mrs_dist_probs_bins.csv',
        index_col='Stroke type')

    # Build the names of the dists that we need from these files
    # by using the input variables:
    dist_pre_stroke_str = 'pre_stroke_' + occlusion_str
    dist_no_treatment_str = 'no_treatment_' + occlusion_str
    dist_t0_treatment_str = (
        't0_treatment_' + occlusion_str + '_' + treatment_str)
    dist_no_effect_str = (
        'no_effect_' + occlusion_str + '_' + treatment_str + '_deaths')

    try:
        # Get the dists from the data array using the strings:
        dist_pre_stroke = mrs_dists_bins.loc[dist_pre_stroke_str].values
        dist_no_treatment = mrs_dists_bins.loc[dist_no_treatment_str].values
        dist_t0_treatment = mrs_dists_bins.loc[dist_t0_treatment_str].values
        dist_no_effect = mrs_dists_bins.loc[dist_no_effect_str].values

        dist_cumsum_pre_stroke = \
            mrs_dists_cumsum.loc[dist_pre_stroke_str].values
        dist_cumsum_no_treatment = \
            mrs_dists_cumsum.loc[dist_no_treatment_str].values
        dist_cumsum_t0_treatment = \
            mrs_dists_cumsum.loc[dist_t0_treatment_str].values
        dist_cumsum_no_effect = \
            mrs_dists_cumsum.loc[dist_no_effect_str].values
        return [
            dist_pre_stroke, dist_no_treatment,
            dist_t0_treatment, dist_no_effect,
            dist_cumsum_pre_stroke, dist_cumsum_no_treatment,
            dist_cumsum_t0_treatment, dist_cumsum_no_effect
            ]
    except AttributeError:
        err_str = (f':warning: No data for {occlusion_str} and ' +
                   f'{treatment_str.upper()}.')
        return err_str


def find_useful_dist_dict(
        occlusion_input,
        treatment_input,
        time_input,
        time_no_effect_ivt=int(6.3*60),
        time_no_effect_mt=int(8*60)
        ):
    # Use the inputs to set up some strings for importing data:
    occlusion_str = 'nlvo' if 'nLVO' in occlusion_input else 'lvo'
    treatment_str = 'mt' if 'MT' in treatment_input else 'ivt'
    # excess_death_str = '_'+treatment_str+'_deaths'

    time_no_effect = (time_no_effect_ivt if 'ivt' in treatment_str
                      else time_no_effect_mt)

    # ----- Select the mRS data -----
    all_dists = find_mRS_dists_from_file(occlusion_str, treatment_str)
    if type(all_dists) == str:
        # No data was imported, so print a warning message:
        st.warning(all_dists)
        st.stop()
    else:
        dist_pre_stroke, dist_no_treatment, \
            dist_t0_treatment, dist_no_effect, \
            dist_cumsum_pre_stroke, dist_cumsum_no_treatment, \
            dist_cumsum_t0_treatment, dist_cumsum_no_effect = all_dists

    # ----- Find probability with time -----
    (dist_time_input_treatment, dist_cumsum_time_input_treatment,
        A_list, b_list) = \
        find_dists_at_chosen_time(
            dist_cumsum_t0_treatment, dist_cumsum_no_effect,
            time_input, time_no_effect)

    #  ----- Make data frames -----
    # Set up headings for the rows and columns:
    headings_rows = [
        'Pre-stroke',
        'Treatment at 0hr',
        f'Treatment at {time_input//60}hr {time_input%60}min',
        f'Treatment at {time_no_effect//60}hr {time_no_effect%60}min',
        'Not treated'
        ]
    headings_cols_cumsum = [f'mRS<={i}' for i in range(7)]
    headings_cols_bins = [f'mRS={i}' for i in range(7)]

    # Build data frames:
    df_dists_cumsum = pd.DataFrame(
        [
            dist_cumsum_pre_stroke,
            dist_cumsum_t0_treatment,
            dist_cumsum_time_input_treatment,
            dist_cumsum_no_effect,
            dist_cumsum_no_treatment
        ],
        index=headings_rows,
        columns=headings_cols_cumsum
        )
    df_dists_bins = pd.DataFrame(
        [
            dist_pre_stroke,
            dist_t0_treatment,
            dist_time_input_treatment,
            dist_no_effect,
            dist_no_treatment
        ],
        index=headings_rows,
        columns=headings_cols_bins
        )

    # ----- Find cumulative added utility -----
    (mRS_dist_mix, weighted_added_utils, mRS_list_time_input_treatment,
        mRS_list_no_treatment) = find_added_utility_between_dists(
        dist_cumsum_time_input_treatment, dist_cumsum_no_treatment,
        utility_weights
    )

    # Stick everything into one dict:
    output_dict = dict(
        time_no_effect=time_no_effect,
        dist_pre_stroke=dist_pre_stroke,
        dist_no_treatment=dist_no_treatment,
        dist_t0_treatment=dist_t0_treatment,
        dist_no_effect=dist_no_effect,
        dist_cumsum_pre_stroke=dist_cumsum_pre_stroke,
        dist_cumsum_no_treatment=dist_cumsum_no_treatment,
        dist_cumsum_t0_treatment=dist_cumsum_t0_treatment,
        dist_cumsum_no_effect=dist_cumsum_no_effect,
        dist_time_input_treatment=dist_time_input_treatment,
        dist_cumsum_time_input_treatment=dist_cumsum_time_input_treatment,
        A_list=A_list,
        b_list=b_list,
        df_dists_cumsum=df_dists_cumsum,
        df_dists_bins=df_dists_bins,
        mRS_dist_mix=mRS_dist_mix,
        weighted_added_utils=weighted_added_utils,
        mRS_list_time_input_treatment=mRS_list_time_input_treatment,
        mRS_list_no_treatment=mRS_list_no_treatment,
        treatment_time=time_input
    )
    return output_dict
