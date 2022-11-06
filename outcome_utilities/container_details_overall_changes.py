import streamlit as st 

def main(args):
    """
    Main code for filling the container. 
    """
    st.write('For each group, the weighted change is equal to the product '+
                'of the following:')
    st.write('+ proportion with this stroke type (%)')
    st.write('+ proportion receiving this treatment (%)')
    st.write('+ total change across this population')
    st.write('The final change given in the Results section above ' +
                'is the sum of the weighted changes.')

    (prop_dict, 
    mean_mRS_dict_nlvo_ivt_case1,
    mean_mRS_dict_lvo_ivt_case1,
    mean_mRS_dict_lvo_mt_case1,
    mean_mRS_change_case1,
    mean_mRS_dict_nlvo_ivt_case2,
    mean_mRS_dict_lvo_ivt_case2,
    mean_mRS_dict_lvo_mt_case2,
    mean_mRS_change_case2,
    mean_util_dict_nlvo_ivt_case1,
    mean_util_dict_lvo_ivt_case1,
    mean_util_dict_lvo_mt_case1,
    mean_util_change_case1,
    mean_util_dict_nlvo_ivt_case2,
    mean_util_dict_lvo_ivt_case2,
    mean_util_dict_lvo_mt_case2,
    mean_util_change_case2) = args

    st.subheader('__Case 1__: mRS')
    print_change_sums(
        prop_dict, 
        mean_mRS_dict_nlvo_ivt_case1,
        mean_mRS_dict_lvo_ivt_case1,
        mean_mRS_dict_lvo_mt_case1,
        )
    # st.text('Check:' + f'{mean_mRS_change_case1:22.2f}')
    st.text(' ')

    # with cols_metric[0]:
    st.subheader('__Case 1__: Utility')
    print_change_sums(
        prop_dict, 
        mean_util_dict_nlvo_ivt_case1,
        mean_util_dict_lvo_ivt_case1,
        mean_util_dict_lvo_mt_case1,
        util=True
        )
    # st.text('Check:' + f'{mean_util_change_case1:22.3f}')
    st.text(' ')


    # with cols_metric[1]:
    st.subheader('__Case 2__: mRS')
    print_change_sums(
        prop_dict, 
        mean_mRS_dict_nlvo_ivt_case2,
        mean_mRS_dict_lvo_ivt_case2,
        mean_mRS_dict_lvo_mt_case2,
        )
    # st.text('Check:' + f'{mean_mRS_change_case2:22.2f}')
    st.text(' ')


    # with cols_metric[1]:
    st.subheader('__Case 2__: Utility')
    print_change_sums(
        prop_dict, 
        mean_util_dict_nlvo_ivt_case2,
        mean_util_dict_lvo_ivt_case2,
        mean_util_dict_lvo_mt_case2,
        util=True
        )
    # st.text('Check:' + f'{mean_util_change_case2:22.3f}')



    # prop_dict = dict(
    #     nlvo = prop_nlvo,
    #     lvo = prop_lvo,
    #     ich = prop_ich,
    #     nlvo_treated_ivt_only = prop_nlvo_treated_ivt_only,
    #     lvo_treated_ivt_only = prop_lvo_treated_ivt_only,
    #     lvo_treated_ivt_mt = prop_lvo_treated_ivt_mt,
    #     ich_treated = prop_ich_treated,
    #     lvo_mt_also_receiving_ivt = prop_lvo_mt_also_receiving_ivt,
    #     treated_population = treated_population
    # )




def print_change_sums(
    prop_dict, 
    mean_dict_nlvo_ivt,
    mean_dict_lvo_ivt,
    mean_dict_lvo_mt,
    util=False
    ):

    cumulative_changes = 0.0
    big_p_str = r'''\begin{align*}'''
    # Add column headings:
    big_p_str += (r'''& & \mathrm{Proportion} & & \mathrm{Proportion} & & \mathrm{Weighted} \\
                    ''')
    big_p_str += (r'''& & \mathrm{with\ type} & & \mathrm{treated} & & \mathrm{change}  \\
                    ''')

    big_p_str, outcome_total, outcome_total_modified = \
        build_latex_combo_change_string(
            prop_dict['nlvo'],
            prop_dict['nlvo_treated_ivt_only'],
            mean_dict_nlvo_ivt['diff_no_treatment'],
            util,
            r'''\mathrm{nLVO\ with\ IVT:}''',
            big_p_str
        )
    cumulative_changes+=outcome_total


    big_p_str, outcome_total, outcome_total_modified = \
        build_latex_combo_change_string(
            prop_dict['lvo'],
            prop_dict['lvo_treated_ivt_only'],
            mean_dict_lvo_ivt['diff_no_treatment'],
            util,
            r'''\mathrm{LVO\ with\ IVT:}''',
            big_p_str
        )
    cumulative_changes+=outcome_total


    big_p_str, outcome_total, outcome_total_modified = \
        build_latex_combo_change_string(
            prop_dict['lvo'],
            prop_dict['lvo_treated_ivt_mt'],
            mean_dict_lvo_mt['diff_no_treatment'],
            util,
            r'''\mathrm{LVO\ with\ MT:}''',
            big_p_str
        )
    cumulative_changes+=outcome_total


    # Add total beneath the rest: 
    # big_p_str += r'''& & & & & & -----\\\\'''
    big_p_str += r'''\hline''' 
    big_p_str += r'''& & & & & & \mathrm{Total}: &'''

    if cumulative_changes>=0:
        # Add sneaky + for alignment
        big_p_str += r'\phantom{+}'

    if util==True:
        big_p_str += f'{cumulative_changes:7.3f}\\\\'
    else:
        big_p_str += f'{cumulative_changes:6.2f}\\\\'
    big_p_str += r'''\end{align*}'''

    st.latex(big_p_str)



def build_latex_combo_change_string(
        prop_type, prop_treated, outcome_change, util=False, row_label='', big_p_str=''):

    outcome_total = prop_type*prop_treated*outcome_change

    # To do: 
    outcome_total_orig = 0.0#np.NaN 

    # Sometimes get values of "-0.0" with the minus sign 
    # and this messes up the formatting. Manually reset it:
    if outcome_total == 0.0:
        outcome_total = 0.0
    if outcome_change == 0.0:
        outcome_change = 0.0

    # Getting some weird behaviour in the rounding, e.g. 
    # >>> f'{0.00750:6.3f}'
    # ' 0.007'
    # >>> f'{0.00850:6.3f}'
    # ' 0.009'
    # Only an issue for values exactly half-way between the rounded
    # point. Best to ignore it... 

    # Add sneaky + for alignment: 
    p_str_change = r'\phantom{+}' if outcome_change>=0 else ''
    p_str_total = r'\phantom{+}' if outcome_total>=0 else ''

    if util==True:
        p_str_change += f'{outcome_change:7.4f}'
        p_str_total += f'{outcome_total:7.4f}'
    else:
        p_str_change += f'{outcome_change:6.3f}'
        p_str_total += f'{outcome_total:6.3f}'

    big_p_str += row_label 
    big_p_str += r''' & &'''
    big_p_str += f'{100*prop_type:2.0f}\%'
    big_p_str += r''' & \times &'''
    big_p_str += f'{100*prop_treated:4.1f}\%'
    big_p_str += r''' & \times & '''
    big_p_str += p_str_change 
    big_p_str += r'''  = & '''
    big_p_str += p_str_total 

    # Next line:
    big_p_str += '\\\\ '

    return big_p_str, outcome_total, outcome_total_orig


# \begin{align*}
# & & \mathrm{Proportion} & & \mathrm{Proportion} & & \mathrm{Weighted} \\ 
# & & \mathrm{with\ type} & & \mathrm{treated} & & \mathrm{change} \\ 
# \mathrm{nLVO\ with\ IVT} & 65% & \times &15.5% & \times & \phantom{+} 0.102 & = \phantom{+} 0.010\\ 
# \mathrm{LVO\ with\ IVT:} & 35% & \times & 0.0% & \times & \phantom{+} 0.032 & = \phantom{+} 0.000\\ 
# \mathrm{LVO\ with\ MT}   & 35% & \times &28.6% & \times & \phantom{+} 0.075 & = \phantom{+} 0.007\\ 
# \hline
# & & & & & & \mathrm{Total}: &\phantom{+}0.018\\
# \end{align*}