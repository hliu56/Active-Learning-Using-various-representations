import pandas as pd
import numpy as np
from sklearn import preprocessing



def get_input(myfile):
    myfile =r'Data/CombinedPSP.csv'
    df_load = pd.read_csv(myfile)

    df_load['JSC']=df_load['JSC'].abs()
    df_reduce=df_load.iloc[:,3:]
    df_refine=df_reduce.iloc[:,np.r_[0:3,4:23]]

    x = df_refine.values #returns a numpy array
    standard_scaler = preprocessing.StandardScaler()
    x_scaled = standard_scaler.fit_transform(x)
    df_refine_standardize = pd.DataFrame(x_scaled)
    df_refine_standardize.columns=df_refine.columns
    df_refine_standardize

    X = df_refine_standardize.loc[:,['STAT_e', 'DISS_f10_D', 'CT_n_D_adj_An', 'CT_n_A_adj_Ca', 'DISS_wf10_D']]    
    y = df_refine_standardize.loc[:,['JSC']]

    return X, y