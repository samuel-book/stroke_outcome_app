import pandas as pd

def make_combo_mRS_bin_dataframe(df1, df2, treatment_time1, treatment_time2):
    if treatment_time1<treatment_time2:
        df_main = df1 
        df_extra = df2 
    elif treatment_time2<treatment_time1:
        df_main = df2 
        df_extra = df1 
    else:
        # Same rows in both so just return one: 
        return df1 

    new_df = pd.concat((df_main.iloc[:3], df_extra.iloc[2:3], df_main.iloc[3:]), axis=0)
    return new_df 