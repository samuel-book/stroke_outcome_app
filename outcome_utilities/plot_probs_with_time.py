import numpy as np 
import matplotlib.pyplot as plt
import streamlit as st

#@st.cache
def plot_probs_filled(A,b,times_mins, colour_list=[], 
    ax=None, title=''):
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

    ax.legend(loc='center left', bbox_to_anchor=[1.0,0.5,0.2,0.2], 
        title='mRS')
    ax.set_ylabel('Probability')
    ax.set_xlabel('Onset to treatment time (hours)')

    ax.set_ylim(0, 1)
    ax.set_xlim(times_hours[0],times_hours[-1])

    ax.set_xticks(np.arange(times_hours[0],times_hours[-1]+0.01,1))
    ax.set_xticks(np.arange(times_hours[0],times_hours[-1]+0.01,0.25), 
        minor=True)

    ax.set_yticks(np.arange(0,1.01,0.2))
    ax.set_yticks(np.arange(0,1.01,0.05), minor=True)
    ax.tick_params(top=True, bottom=True, left=True, right=True, which='both')

    ax.set_title(title)