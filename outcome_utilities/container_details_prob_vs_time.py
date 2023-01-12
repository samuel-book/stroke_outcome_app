import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

from .fixed_params import colour_list, time_no_effect_mt
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
        time_no_effect, 
        A_list,
        b_list,
        colour_list,
        treatment_times,
        treatment_labels=[],
        time_no_effect_mt=8*60
        ):
    # --- Plot probability with time -----
    times_to_plot = np.arange(0, time_no_effect + 1e-7, 5)
    times_hours = times_to_plot / 60.0

    # P(mRS<=5)=1.0 at all times, so it has no defined A, a, and b.
    # Instead append to this array a 0.0, which won't be used directly
    # but will allow the "for" loop to go round one extra time.
    A_list = np.append(A_list, 0.0)
    cum_probs_with_time_lists = []
    for i, A_i in enumerate(A_list):
        # Define the probability line, p_i:
        if i < 6:
            p_i = 1.0/(1.0 + np.exp(-A_i - b_list[i]*times_to_plot))
        else:
            # P(mRS<=5)=1.0 at all times:
            p_i = np.full(times_to_plot.shape, 1.0)
        cum_probs_with_time_lists.append(p_i)


    # Convert cumulative hazard lists into non-cumulative
    # for easier plotting with plotly.
    probs_with_time_lists = [cum_probs_with_time_lists[0]]
    for mRS in np.arange(1, 7):
        # For each mRS, subtract the values that came before it.
        diff_list = np.array(cum_probs_with_time_lists[mRS]-cum_probs_with_time_lists[mRS-1])
        # # Ready to delete (15th Dec 2022):
        # # Attempted fix for weird mRS 5 line for age > 83 or so.
        # # If any difference is negative, set it to zero:
        # diff_list[np.where(diff_list < 0)] = 0.0
        probs_with_time_lists.append(diff_list)

    # Build this data into a big dataframe for plotly.
    # It wants each row in the table to have [mRS, year, hazard].
    for i in range(7):
        # The symbol for less than / equal to: ≤
        mRS_list = [  # 'mRS='+f'{i}'
            f'{i}' for t in times_hours]
        # Use dtype=object to keep the mixed strings (mRS),
        # integers (years) and floats (hazards).
        data_here = np.transpose(
            np.array([mRS_list, times_hours, probs_with_time_lists[i], cum_probs_with_time_lists[i]],
                     dtype=object)
            )

        if i == 0:
            # Start a new big array that will store all the data:
            data_to_plot = data_here
        else:
            # Add this data to the existing big array:
            data_to_plot = np.vstack((data_to_plot, data_here))

    # Pop this data into a dataframe:
    df_to_plot = pd.DataFrame(
        data_to_plot,
        columns=['mRS', 'Time (hours)', 'Probability', 'Cumulative probability']
        )

    # Plot the data:
    fig = px.area(
        df_to_plot,
        x='Time (hours)', y='Probability', color='mRS',
        custom_data=['Cumulative probability', 'mRS'],
        color_discrete_sequence=colour_list
        )
    # The custom_data aren't directly plotted in the previous lines,
    # but are loaded ready for use with the hover template later.

    # Set axis labels:
    fig.update_xaxes(title_text='Time since onset (hours)')
    fig.update_yaxes(title_text='Cumulative probability (%)')
    # fig.update_layout(legend_title='mRS', title_x=0.5)

    # Hover settings:
    # When hovering, highlight all mRS bins' points for chosen x:
    fig.update_layout(hovermode='x unified')
    # Remove default bulky hover messages:
    fig.update_traces(hovertemplate=None)
    # I don't know why, but the line with <extra></extra> is required
    # to remove the default hover label before the rest of this.
    # Otherwise get "0 mRS=0 ..."
    fig.update_traces(
        hovertemplate=(
            # 'mRS=%{customdata[1]}: %{y:>6.2f}' +
            # 5 * '\U00002002' +
            'mRS≤%{customdata[1]}: %{customdata[0]:4.2f}' +
            '<extra></extra>'
            )
        )
    
    # # Figure title:
    # fig.update_layout(title_text='Hazard function for Death by mRS',
    #                   title_x=0.5)
    # Change axis:
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(range=[0, time_no_effect_mt/60],
                     constrain='domain')  # For aspect ratio.
    # Update ticks:
    # fig.update_xaxes(tick0=0, dtick=5)
    # fig.update_yaxes(tick0=0, dtick=10)


    # Add vertical line at treatment times.

    # treatment_times,
    # treatment_labels=[],
    for i, treatment_time in enumerate(treatment_times):
        fig.add_vline(
            x=treatment_time / 60.0,
            line=dict(color='black', width=2.0),
            # layer='below'  # Puts it below
            )
        fig.add_annotation(
            x=treatment_time / 60.0,
            y=1.0,
            text=treatment_labels[i],
            showarrow=True,
            # yshift=-100,
            ax=0,  # Make arrow vertical - a = arrow, x = x-shift.
            ay=-20,  # Make the label sit above the top of the graph
            textangle=-45
            )

    # # Move legend to side
    # fig.update_layout(legend=dict(
    #     orientation='v', #'h',
    #     yanchor='top',
    #     y=1,
    #     xanchor='left',
    #     x=1.03
    # ))

    # Set aspect ratio:
    fig.update_yaxes(
        scaleanchor='x',
        scaleratio=2.5,
        constrain='domain'
    )
    # Reduce size of figure by adjusting margins:
    fig.update_layout(margin=dict(b=0, t=0), height=250)

    # Write to streamlit:
    st.plotly_chart(fig, use_container_width=True)



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
