import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .fixed_params import colour_list, time_no_effect_mt


def compare_probs_with_time(dict1, dict2):
    # Plot probs with time

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
    # Add in the treatment times and final times:
    times_to_plot = np.sort(np.unique(np.append(
        [*treatment_times, time_no_effect], times_to_plot
    )))

    times_to_plot_m = pd.to_datetime( times_to_plot, unit='m')

    times_hours = times_to_plot / 60.0
    times_hours_str = [
        f'{int(60*t//60):2d}hr ' +
        f'{int(60*t%60):2d}min'
        for t in times_hours]

    treatment_times_str = [
        f'{int(t//60):2d}hr ' +
        f'{int(t%60):2d}min'
        for t in treatment_times]

    x_vals = np.arange(len(times_to_plot))

    # times_str_hrs = [
    #     f'{int(60*t//60):2d}hr '
    #     for t in times_hours]
    # times_str_mins = [
    #     f'{int(60*t%60):2d}min'
    #     for t in times_hours]

    
    # times_hours_d3_str = [
    #     f'{int(60*t//60):2d}:' +  # hours
    #     f'{int(60*t%60):2d}:' +  # minutes
    #     f'{360*t%60}'  # seconds
    #     for t in times_hours]

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

    # # Build this data into a big dataframe for plotly.
    # # It wants each row in the table to have [mRS, year, hazard].
    # for i in range(7):
    #     # The symbol for less than / equal to: ≤
    #     mRS_list = [  # 'mRS='+f'{i}'
    #         f'{i}' for t in times_hours]
    #     # Use dtype=object to keep the mixed strings (mRS),
    #     # integers (years) and floats (hazards).
    #     data_here = np.transpose(
    #         np.array([mRS_list, times_hours, probs_with_time_lists[i], cum_probs_with_time_lists[i]],
    #                  dtype=object)
    #         )

    #     if i == 0:
    #         # Start a new big array that will store all the data:
    #         data_to_plot = data_here
    #     else:
    #         # Add this data to the existing big array:
    #         data_to_plot = np.vstack((data_to_plot, data_here))

    # # Pop this data into a dataframe:
    # df_to_plot = pd.DataFrame(
    #     data_to_plot,
    #     columns=['mRS', 'Time (hours)', 'Probability', 'Cumulative probability']
    #     )

    # # Plot the data:
    # fig = px.area(
    #     df_to_plot,
    #     x='Time (hours)', y='Probability', color='mRS',
    #     custom_data=['Cumulative probability', 'mRS'],
    #     color_discrete_sequence=colour_list
    #     )
    # # The custom_data aren't directly plotted in the previous lines,
    # # but are loaded ready for use with the hover template later.

    fig = go.Figure()
    for i in range(7):
        customdata = np.stack((
            cum_probs_with_time_lists[i],
            [i]*len(times_hours),
            # times_str_hrs,
            # times_str_mins
            ), axis=-1)
        # Line and fill:
        fig.add_trace(go.Scatter(
            x=times_to_plot_m, #x_vals,  #times_hours_str, #times_hours_d3_str,
            y=probs_with_time_lists[i],
            mode='lines',
            line=dict(color=colour_list[i]),
            stackgroup='one',
            name=f'{i}',
            customdata=customdata,
        ))


    # # Update x ticks:
    # fig.update_layout(
    #     xaxis=dict(
    #         tickmode='array',
    #         tickvals=x_vals,
    #         ticktext=times_hours_str
    #     )
    # )

    # # Update x ticks:
    # fig.update_layout(
    #     xaxis=dict(
    #         tickmode='array',
    #         tickvals=times_hours,#[::10],
    #         ticktext=times_hours_d3_str,#[::10],
    #         nticks=10
    #     )
    # )
    # fig.update_xaxes(
    #     minor=dict(
    #         tickmode='array',
    #         tickvals=times_hours,#[::10],
    #         # ticktext=times_hours_d3_str,#[::10],
    #     )
    # )



    # Set axis labels:
    fig.update_xaxes(title_text='Time since onset')
    fig.update_yaxes(title_text='Cumulative probability')
    fig.update_layout(legend_title='mRS')  #, title_x=0.5)

    # Hover settings:
    # When hovering, highlight all mRS bins' points for chosen x:
    fig.update_layout(hovermode='x unified')
    # Remove default bulky hover messages:
    # fig.update_traces(hovertemplate=None)
    # I don't know why, but the line with <extra></extra> is required
    # to remove the default hover label before the rest of this.
    # Otherwise get "0 mRS=0 ..."
    fig.update_traces(
        hovertemplate=(
            # 'mRS=%{customdata[1]}: %{y:>6.2f}' +
            # 5 * '\U00002002' +
            'mRS≤%{customdata[1]}: %{customdata[0]:6.4f}' +
            '<extra>' +
            # '%{customdata[2]} %{customdata[3]}' +
            '</extra>'
            )
    )

    # fig.update_layout(
    #     xaxis_tickformat='%H%M',
    #     xaxis_hoverformat='%H%M'  # '%{customdata[2]} %{customdata[3]}'
    # )

    # fig.update_xaxes(
    #     tickformat='%H%M',
    #     hoverformat='%H%M'  # '%{customdata[2]} %{customdata[3]}'
    # )


    # # Figure title:
    # fig.update_layout(title_text='Hazard function for Death by mRS',
    #                   title_x=0.5)
    # Change axis:

    time_no_effect_mt_str = (
        f'{int(time_no_effect_mt//60):2d}hr ' +
        f'{int(time_no_effect_mt%60):2d}min'
    )
    
    # Secret bonus line to help with the range:
    # all_times = np.arange(0, time_no_effect_mt + 1e-7, 5) / 60.0
    all_times = np.array([0.0, time_no_effect_mt]) / 60.0
    all_times_m = pd.to_datetime(all_times, unit='m')

    all_times_hours_str = [
        f'{int(60*t//60):2d}hr ' +
        f'{int(60*t%60):2d}min'
        for t in all_times]

    # fig.add_trace(go.Scatter(
    #     x=all_times_m, # np.arange(len(all_times_hours_str)),
    #     y=[-1]*len(all_times),
    #     showlegend=False,
    #     hoverinfo='skip'
    # ))


    # # Update x ticks:
    # all_x_vals = np.arange(len(all_times_hours_str))
    # fig.update_layout(
    #     xaxis=dict(
    #         tickmode='array',
    #         tickvals=all_x_vals,
    #         ticktext=all_times_hours_str
    #     )
    # )

    x_min = pd.to_datetime(0.0, unit='m')
    x_max = pd.to_datetime(time_no_effect_mt, unit='m')

    fig.update_xaxes(range=[x_min, x_max], #time_no_effect_mt/60],
                     constrain='domain')  # For aspect ratio.
    # fig.update_xaxes(range=[0, time_no_effect_mt], #time_no_effect_mt/60],
    #                  constrain='domain')  # For aspect ratio.
    # fig.update_xaxes(range=['0hr 0min', time_no_effect_mt_str], #time_no_effect_mt/60],
    #                  constrain='domain')  # For aspect ratio.

    # Make the y-axis max a bit bigger than 1 to make sure the label
    # at 1 is shown.
    fig.update_yaxes(range=[0, 1 + 1e-2])
    # Update ticks:
    # Reduce the number of x-axis ticks shown.
    # dtick=12 means every 12th tick is shown, i.e. one per hour.
    # st.write([f'{i}' for i in np.arange(np.ceil(time_no_effect_mt/60.0)+1)])
    # fig.update_xaxes(tick0=0, dtick=12)#, ticktext=[f'{i}' for i in np.arange(np.ceil(time_no_effect_mt/60.0)+1)])
    # fig.update_yaxes(tick0=0, dtick=10)



    # # Update x ticks:
    # fig.update_layout(
    #     xaxis=dict(
    #         tickmode='array',
    #         # tickvals=all_times_hours_str[::12],
    #         tickvals=all_x_vals[::12],
    #         ticktext=[f'{int(i)}hr' for i in np.arange(np.ceil(time_no_effect_mt/60.0)+1)],#[::10],
    #     )
    # )

    # Add vertical line at treatment times.
    for i, treatment_time in enumerate(treatment_times):
        
        treatment_time_m = pd.to_datetime(treatment_time, unit='m')

        fig.add_vline(
            x=treatment_time_m,
            line=dict(color='black', width=2.0)
            )
        fig.add_annotation(
            x=treatment_time_m,
            y=1.0,
            text=treatment_labels[i],
            showarrow=True,
            arrowhead=0,
            # yshift=-100,
            ax=0,    # Make arrow vertical - a = arrow, x = x-shift.
            ay=-20,  # Make the label sit above the top of the graph
            textangle=-45
            )


    fig.update_xaxes(
        # dtick="M1",
        tickformat="%H %M"
    )

    # Remove grid lines:
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    # Set aspect ratio:
    fig.update_yaxes(
        scaleanchor='x',
        scaleratio=2.5,
        constrain='domain'
    )
    # Reduce size of figure by adjusting margins:
    fig.update_layout(margin=dict(b=0, t=30), height=250)

    # Write to streamlit:
    st.plotly_chart(fig, use_container_width=True)


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
