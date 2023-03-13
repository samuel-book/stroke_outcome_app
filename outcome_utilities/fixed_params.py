import numpy as np
import streamlit as st

# Imports used for defining default colours:
# import plotly.express as px
# import matplotlib.pyplot as plt


def page_setup():
    # The following options set up the display in the tab in your
    # browser.
    # Set page to widescreen must be first call to st.
    st.set_page_config(
        page_title='Population stroke outcomes',
        page_icon='📋',
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


# Define built-in colours for mRS bands:
# # Change default colour scheme:
# plt.style.use('seaborn-colorblind')
# colour_list = make_colour_list()
# Colours as of 16th January 2023:
# (the first six are from seaborn-colorblind)
colour_list = [
    "#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#56B4E9",
    "DarkSlateGray"  # mRS=6
    ]

# Define built-in colours for IVT and MT labels:
# plotly_colours = px.colors.qualitative.Plotly
# Colours as of 16th January 2023:
plotly_colours = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
    ]

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
    time_to_ivt=':syringe:',
    time_to_mt=':wrench:'
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
    time_to_ivt='\U0001f489',
    time_to_mt='\U0001f527'
)

# Other emoji: 💊

# Emoji unicode reference:
# 🔧 \U0001f527
# 🏥 \U0001f3e5
# 🚑 \U0001f691
# 💉 \U0001f489
