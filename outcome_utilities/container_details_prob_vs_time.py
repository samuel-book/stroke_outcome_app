import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .fixed_params import make_fig_legend, colour_list, time_no_effect_mt
from .plot_probs_with_time import plot_probs_filled
from outcome_utilities.inputs import write_text_from_file


def main(
        nlvo_ivt_case1_dict,
        nlvo_ivt_case2_dict,
        lvo_ivt_case1_dict,
        lvo_ivt_case2_dict,
        lvo_mt_case1_dict,
        lvo_mt_case2_dict
        ):

    # ----- Add legend -----
    fig_legend = make_fig_legend(colour_list)
    st.pyplot(fig_legend)

    tab1, tab2, tab3, tab4 = st.tabs([
        'nLVO treated with IVT',
        'LVO treated with IVT only',
        'LVO treated with MT',
        'ICH'
    ])

    with tab1:
        compare_probs_with_time(nlvo_ivt_case1_dict, nlvo_ivt_case2_dict)
    with tab2:
        compare_probs_with_time(lvo_ivt_case1_dict, lvo_ivt_case2_dict)
    with tab3:
        compare_probs_with_time(lvo_mt_case1_dict, lvo_mt_case2_dict)
    with tab4:
        st.write('Nothing to see here.')


def compare_probs_with_time(dict1, dict2):
    # Plot probs with time

    write_text_from_file('pages/text_for_pages/2_Probs_with_time.txt',
                         head_lines_to_skip=3)

    do_probs_with_time(
        dict1['time_no_effect'], dict1['A_list'], dict2['b_list'],
        colour_list,
        [dict1['treatment_time'], dict2['treatment_time']],
        treatment_labels=[f'Case {i+1}' for i in range(2)],
        time_no_effect_mt=time_no_effect_mt
        )

    # # Tabulate mRS bins
    st.subheader('mRS data tables')
    st.write(
        'This table contains the probability distributions ' +
        'at key points from the probability vs. time graph above.'
        )
    df_combo = make_combo_mRS_bin_dataframe(
        dict1['df_dists_bins'], dict2['df_dists_bins'],
        dict1['treatment_time'], dict2['treatment_time'])
    st.table(df_combo)


def do_probs_with_time(
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


def make_combo_mRS_bin_dataframe(df1, df2, treatment_time1, treatment_time2):
    if treatment_time1 < treatment_time2:
        df_main = df1
        df_extra = df2
    elif treatment_time2 < treatment_time1:
        df_main = df2
        df_extra = df1
    else:
        # Same rows in both so just return one:
        return df1

    new_df = pd.concat(
        (df_main.iloc[:3], df_extra.iloc[2:3], df_main.iloc[3:]), axis=0
        )
    return new_df
