import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def page_setup():
    # The following options set up the display in the tab in your
    # browser.
    # Set page to widescreen must be first call to st.
    st.set_page_config(
        page_title='Stroke outcome modelling',
        page_icon=':ambulance:',
        # layout='wide'
        )


def make_colour_list():
    """Define the colours for plotting mRS bins."""
    # Get current matplotlib style sheet colours:
    colour_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Remove extra colours:
    colour_list = colour_list[:6]
    # Add grey for mRS=6 bin:
    colour_list.append('DarkSlateGray')
    # Change to array for easier indexing later:
    colour_list = np.array(colour_list)
    return colour_list


# st.latex(r'''\textcolor{#0072B2}{Hello}''')
# st.latex(r'''\textcolor{#009E73}{Hello}''')
# st.latex(r'''\textcolor{#D55E00}{Hello}''')
# st.latex(r'''\textcolor{#CC79A7}{Hello}''')
# st.latex(r'''\textcolor{#F0E442}{Hello}''')
# st.latex(r'''\textcolor{#56B4E9}{Hello}''')
# st.latex(r'''\textcolor{DarkSlateGray}{Hello}''')


def make_fig_legend(colour_list):
    """Plot a legend for the mRS colours."""
    fig_legend = plt.figure(figsize=(6, 2))
    # Dummy data for legend:
    dummies = []
    for i in range(7):
        dummy = plt.bar(np.NaN, np.NaN, color=colour_list[i], edgecolor='k')
        dummies.append(dummy)

    # Clear to remove automatic blank axis:
    fig_legend.clear()
    # Draw legend using dummy bars:
    fig_legend.legend(
        [*dummies], range(7),
        loc='center', ncol=7, title='mRS colour scheme',
        )
    return fig_legend


# Change default colour scheme:
plt.style.use('seaborn-colorblind')
colour_list = make_colour_list()

utility_weights = np.array(
    [0.97, 0.88, 0.74, 0.55, 0.20, -0.19, 0.00])

# Define maximum treatment times:
time_no_effect_ivt = int(6.3*60)   # minutes
time_no_effect_mt = int(8*60)      # minutes

# Define some emoji for various situations:
emoji_dict = dict(
    # onset=':thumbsdown:',
    onset_to_ambulance_arrival=':ambulance:',
    travel_to_ivt=':hospital:',
    travel_to_mt=':hospital:',
    ivt_arrival_to_treatment=':syringe:',
    transfer_additional_delay=':ambulance:',  # ':hourglass_flowing_sand:',
    travel_ivt_to_mt=':hospital:',
    mt_arrival_to_treatment=':wrench:',
)

emoji_text_dict = dict(
    # onset=':thumbsdown:',
    onset_to_ambulance_arrival='\U0001f691',
    travel_to_ivt='\U0001f3e5',
    travel_to_mt='\U0001f3e5',
    ivt_arrival_to_treatment='\U0001f489',
    transfer_additional_delay='\U0001f691',
    travel_ivt_to_mt='\U0001f3e5',
    mt_arrival_to_treatment='\U0001f527',
)

# Other emoji: 💊

# Emoji unicode reference:
# 🔧 \U0001f527
# 🏥 \U0001f3e5
# 🚑 \U0001f691
# 💉 \U0001f489
