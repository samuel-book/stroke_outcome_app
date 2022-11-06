import streamlit as st 

def main(args):

    (
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
    ) = args
    # Put the two metrics in columns: 
    met_col1, met_col2 = st.columns(2)

    # mRS:
    met_col1.subheader('--- Case 1 ---')

    met_col1.write(
        f'💊 Time to IVT: {case1_time_to_ivt//60}hr {case1_time_to_ivt%60}min.'
        )
    met_col1.write(
        f'💉 Time to MT: {case1_time_to_mt//60}hr {case1_time_to_mt%60}min.'
        )
    met_col1.write('-'*20)

    met_col1.metric('Population mean mRS', 
        f'{mean_mRS_treated_case1:0.2f}', 
        f'{mean_mRS_change_case1:0.2f} from no treatment',
        delta_color='inverse' # A negative difference is good.
        )
    met_col1.write('-'*20)
    met_col1.metric('Population mean utility', 
        f'{mean_util_treated_case1:0.3f}', 
        f'{mean_util_change_case1:0.3f} from no treatment',
        )

    # Utility: 
    met_col2.subheader('--- Case 2 ---')

    met_col2.write(
        f'💊 Time to IVT: {case2_time_to_ivt//60}hr {case2_time_to_ivt%60}min.'
        )
    met_col2.write(
        f'💉 Time to MT: {case2_time_to_mt//60}hr {case2_time_to_mt%60}min.'
        )
    met_col2.write('-'*20)

    met_col2.metric('Population mean mRS', 
        f'{mean_mRS_treated_case2:0.2f}', 
        f'{mean_mRS_change_case2:0.2f} from no treatment',
        delta_color='inverse' # A negative difference is good.
        )
    met_col2.write('-'*20)
    met_col2.metric('Population mean utility', 
        f'{mean_util_treated_case2:0.3f}', 
        f'{mean_util_change_case2:0.3f} from no treatment',
        )