import matplotlib.pyplot as plt 
import numpy as np

def plot_timeline(time_dict, ax=None, y=0, emoji_dict={}):
    label_dict = dict( 
        onset = 'Onset',
        onset_to_ambulance_arrival = 'Travel to\nhospital\nbegins',
        travel_to_ivt = 'Arrive at\nIVT\ncentre', 
        travel_to_mt = 'Arrive at\nIVT+MT\ncentre', 
        ivt_arrival_to_treatment = 'IVT',
        transfer_additional_delay = 'Transfer\nbegins',
        travel_ivt_to_mt = 'Arrive at\nIVT+MT\ncentre',
        mt_arrival_to_treatment = 'MT',
        )

    if ax==None:
        fig, ax = plt.subplots()
    time_cumulative = 0.0
    y_under_offset = -0.05
    y_label_offset = 0.30
    for time_key in time_dict.keys():
        t_min = time_dict[time_key]
        time_cumulative += t_min/60.0

        if 'ivt_arrival_to_treatment' in time_key:
            colour = 'b' 
            write_under=True 
        elif 'mt_arrival_to_treatment' in time_key:
            colour = 'r'
            write_under=True 
        else:
            colour = 'k'
            write_under=False 

        if time_dict[time_key]==0.0 and time_key!='onset':
            x_plot = np.NaN 
        else:
            x_plot = time_cumulative
        ax.scatter(x_plot, y, marker='|', s=200, color=colour)

        ax.annotate(
            label_dict[time_key], xy=(x_plot, y+y_label_offset), 
            rotation=0, color=colour, ha='center', va='bottom')
        if write_under: 
            time_str = (f'{int(60*time_cumulative//60):2d}hr '+
                        f'{int(60*time_cumulative%60):2d}min')
            ax.annotate(
                time_str, xy=(x_plot, y+y_under_offset), color=colour,
                ha='center', va='top', rotation=20)
    ax.plot([0, time_cumulative], [y,y], color='k', zorder=0)


def plot_emoji_on_timeline(ax, emoji_dict, time_dict, y=0, aspect=1.0, xlim=[], ylim=[]):
    """Do this after the timeline is drawn so sizing is consistent."""

    y_emoji_offset = 0.15

    y_span = ylim[1] - ylim[0]
    y_size = 1.5*0.07*y_span #* aspect

    x_span = xlim[1] - xlim[0]
    x_size = 1.5*0.04*x_span #/ aspect

    time_cumulative = 0.0 
    for time_key in time_dict.keys():
        if time_key in emoji_dict.keys(): 
            t_min = time_dict[time_key]
            time_cumulative += t_min/60.0           
            if time_dict[time_key]==0.0 and time_key!='onset':
                x_plot = np.NaN 
            else:
                x_plot = time_cumulative
                
                emoji = emoji_dict[time_key].strip(':')
                # Import from file 
                emoji_image = plt.imread('./emoji/'+emoji+'.png')
                ext_xmin = x_plot - x_size*0.5 
                ext_xmax = x_plot + x_size*0.5 
                ext_ymin = y+y_emoji_offset - y_size*0.5 
                ext_ymax = y+y_emoji_offset + y_size*0.5 
                ax.imshow(emoji_image, 
                          extent=[ext_xmin, ext_xmax, ext_ymin, ext_ymax])
        # ax.annotate(emoji_dict[time_key], xy=(time_cumulative, y+y_emoji_offset))


def make_timeline_plot(ax, time_dicts, emoji_dict={}):
    
    y_step = 1.0
    y_vals = np.arange(0.0, y_step*len(time_dicts), y_step)[::-1]
    for i, time_dict in enumerate(time_dicts):
        plot_timeline(time_dict, ax, y=y_vals[i], emoji_dict=emoji_dict)
    
    xlim = ax.get_xlim()
    ax.set_xticks(np.arange(0, xlim[1], (1+(xlim[1]//6))*(10/60)), minor=True)
    ax.set_xlabel('Time since onset (hours)')

    ax.set_ylim(-0.25, y_step*(len(time_dicts)-0.2))
    ylim = ax.get_ylim()
    ax.set_yticks(y_vals)
    ax.set_yticklabels(
        [f'Case {i+1}' for i in range(len(time_dicts))], fontsize=14)

    aspect = 1.0/(ax.get_data_ratio()*2)
    ax.set_aspect(aspect)
    for i, time_dict in enumerate(time_dicts):
        plot_emoji_on_timeline(ax, emoji_dict, time_dict, y=y_vals[i], 
                               aspect=aspect, xlim=xlim, ylim=ylim)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect(aspect)

    for spine in ['top', 'left', 'right']:
        ax.spines[spine].set_color('w')

