import streamlit as st

from .fixed_params import emoji_text_dict


def main(
        # Case 1:
        mean_mRS_treated_case1,
        mean_mRS_change_case1,
        mean_util_treated_case1,
        mean_util_change_case1,
        case1_time_to_ivt,
        case1_time_to_mt,
        # Case 2:
        mean_mRS_treated_case2,
        mean_mRS_change_case2,
        mean_util_treated_case2,
        mean_util_change_case2,
        case2_time_to_ivt,
        case2_time_to_mt,
        ):

    st.subheader('Case 1')
    draw_metrics(
        case1_time_to_ivt,
        case1_time_to_mt,
        mean_mRS_treated_case1,
        mean_mRS_change_case1,
        mean_util_treated_case1,
        mean_util_change_case1
        )

    # st.write('-'*20)

    st.subheader('Case 2')
    draw_metrics(
        case2_time_to_ivt,
        case2_time_to_mt,
        mean_mRS_treated_case2,
        mean_mRS_change_case2,
        mean_util_treated_case2,
        mean_util_change_case2
        )


def draw_metrics(
        time_to_ivt,
        time_to_mt,
        mean_mRS_treated,
        mean_mRS_change,
        mean_util_treated,
        mean_util_change
        ):
    # Put the  metrics in columns:
    cols = st.columns(3)

    # Column 0: Times
    cols[0].caption('Treatment times')
    cols[0].write(
        emoji_text_dict['ivt_arrival_to_treatment'] +
        f' IVT: {time_to_ivt//60}hr {time_to_ivt%60}min'
        )
    cols[0].write(
        emoji_text_dict['mt_arrival_to_treatment'] +
        f' MT: {time_to_mt//60}hr {time_to_mt%60}min'
        )

    # Column 1: mRS
    cols[1].metric(
        'Population mean mRS',
        f'{mean_mRS_treated:0.2f}',
        f'{mean_mRS_change:0.2f} from no treatment',
        delta_color='inverse'  # A negative difference is good.
        )

    # Column 2: utility
    cols[2].metric(
        'Population mean utility',
        f'{mean_util_treated:0.3f}',
        f'{mean_util_change:0.3f} from no treatment',
        )
