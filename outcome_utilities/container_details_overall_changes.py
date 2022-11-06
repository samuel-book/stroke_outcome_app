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
    st.text('Check:' + f'{mean_mRS_change_case1:22.2f}')
    st.text(' ')

    # with cols_metric[1]:
    st.subheader('__Case 2__: mRS')
    print_change_sums(
        prop_dict, 
        mean_mRS_dict_nlvo_ivt_case2,
        mean_mRS_dict_lvo_ivt_case2,
        mean_mRS_dict_lvo_mt_case2,
        )
    st.text('Check:' + f'{mean_mRS_change_case2:22.2f}')
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
    st.text('Check:' + f'{mean_util_change_case1:22.3f}')

    # with cols_metric[1]:
    st.subheader('__Case 2__: Utility')
    print_change_sums(
        prop_dict, 
        mean_util_dict_nlvo_ivt_case2,
        mean_util_dict_lvo_ivt_case2,
        mean_util_dict_lvo_mt_case2,
        util=True
        )
    st.text('Check:' + f'{mean_util_change_case2:22.3f}')



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




def add_population_maths_to_string(b1, b2, b3, util=False, calc_str=''):
    p1 = b1*b2*b3 

    calc_str += f'{100*b1:5.0f}%' + ' X '#r' $\times$ ' 
    calc_str += f'{100*b2:5.1f}%' + ' X '#r' $\times$ ' 

    if util==False:
        calc_str += f'{b3:5.2f}' + ' = '#r' $=$ '
        calc_str += f'{p1:5.2f}'

    else:
        calc_str += f'{b3:5.3f}' + ' = '#r' $=$ '
        calc_str += f'{p1:5.3f}'

    p2=0 
    return calc_str, p1, p2


def print_change_sums(
    prop_dict, 
    mean_dict_nlvo_ivt,
    mean_dict_lvo_ivt,
    mean_dict_lvo_mt,
    util=False
    ):
    sumcheck=0
    calc_str_case1, p1,p2 = add_population_maths_to_string(
        prop_dict['nlvo'],
        prop_dict['nlvo_treated_ivt_only'],
        mean_dict_nlvo_ivt['diff_no_treatment'],
        util
        )
    sumcheck+=p1
    calc_str_case1 += '  (nLVO with IVT)'
    st.text(calc_str_case1)

    calc_str_case1, p1,p2 = add_population_maths_to_string(
        prop_dict['lvo'],
        prop_dict['lvo_treated_ivt_only'],
        mean_dict_lvo_ivt['diff_no_treatment'],
        util
        )
    sumcheck+=p1
    calc_str_case1 += '  (LVO with IVT)'
    st.text(calc_str_case1)

    calc_str_case1, p1, p2 = add_population_maths_to_string(
        prop_dict['lvo'],
        prop_dict['lvo_treated_ivt_mt'],
        mean_dict_lvo_mt['diff_no_treatment'],
        util
        )
    sumcheck+=p1
    calc_str_case1 += '  (LVO with MT)'
    st.text(calc_str_case1)

    st.text('-'*28)
    if util==False:
        st.text('Total:' + f'{sumcheck:22.2f}')
    else:
        st.text('Total:' + f'{sumcheck:22.3f}')
    
