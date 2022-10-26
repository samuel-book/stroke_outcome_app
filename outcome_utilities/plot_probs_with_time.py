import numpy as np 
import matplotlib.pyplot as plt
import streamlit as st

#@st.cache
# @st.experimental_singleton
def plot_probs_filled(A,b,times_mins, colour_list=[], 
    treatment_times=[], treatment_labels=[], 
    ax=None, title='', xmax=None):
    if ax==None:
        ax = plt.subplot()
    # P(mRS<=5)=1.0 at all times, so it has no defined A, a, and b.
    # Instead append to this array a 0.0, which won't be used directly
    # but will allow the "for" loop to go round one extra time.
    A = np.append(A,0.0)
    times_hours = times_mins/60.0

    for i,A_i in enumerate(A):
        if len(colour_list)>0:
            colour = colour_list[i]
        else:
            colour=None
        # Define the probability line, p_i:
        if i<6:
            p_i = 1.0/(1.0 + np.exp(-A_i -b[i]*times_mins)) 
        else:
            # P(mRS<=5)=1.0 at all times:
            p_i = np.full(times_mins.shape,1.0)
            
        # Plot it as before and store the colour used:
        ax.plot(times_hours, p_i, #color=colour)#, label = f'mRS <= {i}')
            color='k', linewidth=1)
        # Fill the area between p_i and the line below, p_j.
        # This area marks where mRS <= the given value.
        # If p_j is not defined yet (i=0), set all p_j to zero:
        p_j = p_j if i>0 else np.zeros_like(p_i)
        ax.fill_between(times_hours, p_i, p_j, label=f'{i}',
                         color=colour)#, alpha=0.3 )
        # ^ alpha is used as a quick way to lighten the fill colour.

        # Store the most recently-created line for the next loop:
        p_j = p_i

    if len(treatment_times)>0:
        if len(treatment_labels)!=len(treatment_times):
            treatment_labels = ['Treatment' for t in treatment_times]
        for i, time_input in enumerate(treatment_times):
            ax.axvline(time_input/60.0, color='k', linestyle=':')
            ax.annotate('|',
                xy=(time_input/60.0, 0.95), va='bottom', ha='center', color='r',
                fontsize=20, zorder=0
                )
            ax.annotate(treatment_labels[i]+'\n',
                xy=(time_input/60.0, 1.0), va='bottom', ha='left', color='r',
                rotation=30
                )



    # ax.legend(loc='upper center', bbox_to_anchor=[0.5,1.2], 
    #     title='mRS', ncol=7)
    ax.set_ylabel('Probability')
    ax.set_xlabel('Onset to treatment time (hours)')

    ax.set_ylim(0, 1)

    if xmax==None:
        xmax = times_hours[-1]

    # Secret annotation to make sure axis doesn't resize when time input
    # is equal to max time:
    ax.annotate('Treatment',
        xy=(xmax, 0.0), va='top', ha='center', color='None'
        )

    ax.set_xlim(times_hours[0],xmax)

    ax.set_xticks(np.arange(times_hours[0],xmax+0.01,1))
    ax.set_xticks(np.arange(times_hours[0],xmax+0.01,0.25), 
        minor=True)

    ax.set_yticks(np.arange(0,1.01,0.2))
    ax.set_yticks(np.arange(0,1.01,0.05), minor=True)
    ax.tick_params(top=True, bottom=True, left=True, right=True, which='both')

    ax.set_title(title)



def find_dists_at_chosen_time(dist_cumsum_t0_treatment, 
    dist_cumsum_no_effect, time_input, time_no_effect):
    """
    Find the mRS distributions at the input treatment time.
    Finds the cumulative and non-cumulative probability distributions.

    Return the A and b lists for later plotting. 
    """
    a_list, b_list, A_list = find_mrs_constants(
        dist_cumsum_t0_treatment, dist_cumsum_no_effect, time_no_effect)
    # The probability P at an arbitrary treatment time is: 
    # P(treatment_time) = 1.0/(1.0 + np.exp(-A -b*treatment_time)) 
    # where the values of a, b, and A are different for each mRS. 


    # ----- Find distribution at chosen treatment time -----
    # Calculate the cumulative probability bins at the chosen time: 
    if time_input > time_no_effect:
        time_input = time_no_effect 
    dist_cumsum_time_input_treatment = (
        find_mrs_bins_t(A_list, b_list, time_input))
    # Append the value for mRS=6:
    dist_cumsum_time_input_treatment = (
        np.append(dist_cumsum_time_input_treatment, 1.0))
    # Make the non-cumulative bins: 
    dist_time_input_treatment = (
        np.diff(dist_cumsum_time_input_treatment, prepend=0.0))
    return (dist_time_input_treatment, dist_cumsum_time_input_treatment,
        A_list, b_list)