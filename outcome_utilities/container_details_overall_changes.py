import streamlit as st


def main(
        prop_dict,
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
        mean_util_change_case2
        ):
    """
    Main code for filling the container.
    """
    st.markdown(
        '''
        In the following sums, the groups always appear in this order:
        + nLVO treated with IVT
        + LVO treated with IVT
        + LVO treated with MT

        For each group, the weighted change is equal to the product
        of the following:
        + proportion with this stroke type (%)
        + proportion receiving this treatment (%)
        + total change across this population
        The final change given in the Results section above
        is the sum of the weighted changes.
        '''
        )

    st.subheader('__Case 1__')
    cols_case1 = st.columns(2)
    with cols_case1[0]:
        st.write('Change in mRS:')
        print_change_sums(
            prop_dict,
            mean_mRS_dict_nlvo_ivt_case1,
            mean_mRS_dict_lvo_ivt_case1,
            mean_mRS_dict_lvo_mt_case1,
            )
        # st.text('Check:' + f'{mean_mRS_change_case1:22.2f}')
    # st.text(' ')

    with cols_case1[1]:
        st.write('Change in utility:')
        print_change_sums(
            prop_dict,
            mean_util_dict_nlvo_ivt_case1,
            mean_util_dict_lvo_ivt_case1,
            mean_util_dict_lvo_mt_case1,
            util=True
            )
        # st.text('Check:' + f'{mean_util_change_case1:22.3f}')
    st.text(' ')

    st.subheader('__Case 2__')
    cols_case2 = st.columns(2)
    with cols_case2[0]:
        st.write('Change in mRS:')
        print_change_sums(
            prop_dict,
            mean_mRS_dict_nlvo_ivt_case2,
            mean_mRS_dict_lvo_ivt_case2,
            mean_mRS_dict_lvo_mt_case2,
            )
        # st.text('Check:' + f'{mean_mRS_change_case2:22.2f}')
    # st.text(' ')

    with cols_case2[1]:
        st.write('Change in utility:')
        print_change_sums(
            prop_dict,
            mean_util_dict_nlvo_ivt_case2,
            mean_util_dict_lvo_ivt_case2,
            mean_util_dict_lvo_mt_case2,
            util=True
            )


def print_change_sums(
        prop_dict,
        mean_dict_nlvo_ivt,
        mean_dict_lvo_ivt,
        mean_dict_lvo_mt,
        util=False
        ):

    cumulative_changes = 0.0
    big_p_str = r'''\begin{align*}'''
    # # Add column headings:
    # big_p_str += (
    #     r''' & & \mathrm{Proportion} & & \mathrm{Proportion} & &''' +
    #     r''' \mathrm{Weighted} \\'''
    #     )
    # big_p_str += (
    #     r''' & & \mathrm{with\ type} & & \mathrm{treated} & &''' +
    #     r''' \mathrm{change}  \\'''
    #     )

    big_p_str, outcome_total, outcome_total_modified = \
        build_latex_combo_change_string(
            prop_dict['nlvo'],
            prop_dict['nlvo_treated_ivt_only'],
            mean_dict_nlvo_ivt['diff_no_treatment'],
            util,
            '',  # r'''\mathrm{nLVO\ with\ IVT:}''',
            big_p_str
            )
    cumulative_changes += outcome_total

    big_p_str, outcome_total, outcome_total_modified = \
        build_latex_combo_change_string(
            prop_dict['lvo'],
            prop_dict['lvo_treated_ivt_only'],
            mean_dict_lvo_ivt['diff_no_treatment'],
            util,
            '',  # r'''\mathrm{LVO\ with\ IVT:}''',
            big_p_str
            )
    cumulative_changes += outcome_total

    big_p_str, outcome_total, outcome_total_modified = \
        build_latex_combo_change_string(
            prop_dict['lvo'],
            prop_dict['lvo_treated_ivt_mt'],
            mean_dict_lvo_mt['diff_no_treatment'],
            util,
            '',  # r'''\mathrm{LVO\ with\ MT:}''',
            big_p_str
        )
    cumulative_changes += outcome_total

    # Add total beneath the rest:
    big_p_str += r'''\hline'''
    big_p_str += r'''& & & & & & \mathrm{Total}: &'''

    if cumulative_changes >= 0:
        # Add sneaky + for alignment
        big_p_str += r'''\phantom{+}'''

    if util is True:
        big_p_str += f'{cumulative_changes:6.3f}\\\\'
    else:
        big_p_str += f'{cumulative_changes:5.2f}\\\\'
    big_p_str += r'''\end{align*}'''

    st.latex(big_p_str)


def build_latex_combo_change_string(
        prop_type,
        prop_treated,
        outcome_change,
        util=False,
        row_label='',
        big_p_str=''
        ):

    outcome_total = prop_type*prop_treated*outcome_change

    # Round the total of the line to the same number of digits
    # as the printed total of the column, otherwise the sums
    # look incorrect.
    if util is True:
        outcome_change = round(outcome_change, 3)
        outcome_total = round(outcome_total, 3)
    else:
        outcome_change = round(outcome_change, 2)
        outcome_total = round(outcome_total, 2)

    # To do:
    outcome_total_orig = 0.0  # np.NaN

    # Sometimes get values of "-0.0" with the minus sign
    # and this messes up the formatting. Manually reset it:
    if outcome_total == 0.0:
        outcome_total = 0.0
    if outcome_change == 0.0:
        outcome_change = 0.0

    # Add sneaky + for alignment:
    p_str_change = r'\phantom{+}' if outcome_change >= 0 else ''
    p_str_total = r'\phantom{+}' if outcome_total >= 0 else ''

    if util is True:
        p_str_change += f'{outcome_change:6.3f}'
        p_str_total += f'{outcome_total:6.3f}'
    else:
        p_str_change += f'{outcome_change:5.2f}'
        p_str_total += f'{outcome_total:5.2f}'

    big_p_str += row_label
    big_p_str += r''' & &'''
    big_p_str += f'{100*prop_type:2.0f}'
    big_p_str += r'''\% & \times &'''
    big_p_str += f'{100*prop_treated:4.1f}'
    big_p_str += r'''\% & \times & '''
    big_p_str += p_str_change
    big_p_str += r'''  = & '''
    big_p_str += p_str_total

    # Next line:
    big_p_str += '\\\\ '

    return big_p_str, outcome_total, outcome_total_orig
