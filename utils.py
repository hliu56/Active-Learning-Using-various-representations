import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing
import sklearn.gaussian_process as gp
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


def normalized_data(df):
    x = df.values #returns a numpy array
    standard_scaler = preprocessing.StandardScaler()
    x_scaled = standard_scaler.fit_transform(x)
    df_refine_standardize = pd.DataFrame(x_scaled)
    df_refine_standardize.columns=df.columns
    return df_refine_standardize


def get_input(myfile):
    '''
    Get data with knowing salient features
    '''
    df_load = pd.read_csv(myfile)

    df_load['JSC']=df_load['JSC'].abs()
    df_reduce=df_load.iloc[:,3:]
    df_refine=df_reduce.iloc[:,np.r_[0:3,4:23]]
    # x = df_refine.values #returns a numpy array
    # standard_scaler = preprocessing.StandardScaler()
    # x_scaled = standard_scaler.fit_transform(x)
    # df_refine_standardize = pd.DataFrame(x_scaled)
    # df_refine_standardize.columns=df_refine.columns
    df_refine_standardize = normalized_data(df_refine)

    X = df_refine_standardize.loc[:,['STAT_e', 'DISS_f10_D', 'CT_n_D_adj_An', 'CT_n_A_adj_Ca', 'DISS_wf10_D']]    
    y = df_refine_standardize.loc[:,['JSC']]

    return X, y

def get_input_all(myfile):
    '''
    Get all data without knowing salient features
    '''
    df_load = pd.read_csv(myfile)

    df_load['JSC']=df_load['JSC'].abs()
    df_reduce=df_load.iloc[:,3:]
    df_refine=df_reduce.iloc[:,np.r_[0:3,4:23]]
    
    df_refine_standardize = normalized_data(df_refine)

    X = df_refine_standardize.iloc[:,1:]    
    y = df_refine_standardize.iloc[:,0:1]

    return X, y


# Hleper funcitons
def computeR2(dataL, X_test, y_test, fs=False):
    '''
    Calculating model performance for testing dataset.

    Parameters:
        dataL (numpy.ndarray): The input dataset to calculate model performance.
        X_test (int): The input data set for testing.
        y_test (str): The output data set for testing.

    Returns:
        r2: the R2 for the model.
        gpr: the Gaussian process regression model.
        MSE: mean square error.
        MAE: mean absolute error.
    '''
    if fs:
        y_trainGP = dataL.iloc[:,-1]
        X_trainGP = dataL.iloc[:,0:-1]
        kernel = Matern(length_scale=1.0)
        gpr=gp.GaussianProcessRegressor(kernel=kernel, random_state=99, n_restarts_optimizer=10)
        gpr.fit(X_trainGP,y_trainGP)
        y_pred,sigma = gpr.predict(X_test, return_std=True) 
        r2 = metrics.r2_score(y_test, y_pred)
        MSE = metrics.mean_squared_error(y_test, y_pred)
        MAE = metrics.mean_absolute_error(y_test, y_pred)
    else:
        y_trainGP = dataL[:,-1]
        X_trainGP = dataL[:,0:-1]
        kernel = Matern(length_scale=1.0)
        gpr=gp.GaussianProcessRegressor(kernel=kernel, random_state=99, n_restarts_optimizer=10)
        gpr.fit(X_trainGP,y_trainGP)
        y_pred,sigma = gpr.predict(X_test, return_std=True) 
        r2 = metrics.r2_score(y_test, y_pred)
        MSE = metrics.mean_squared_error(y_test, y_pred)
        MAE = metrics.mean_absolute_error(y_test, y_pred)

    return np.array([[r2]]), gpr, np.array([[MSE]]),np.array([[MAE]])

def computeR2_train(dataL, X_train, y_train, fs=False):
    '''
    Calculate the model performance for the initial 80% training data set
    '''
    if fs:
        y_trainGP = dataL.iloc[:,-1]
        X_trainGP = dataL.iloc[:,0:-1]
        kernel = Matern(length_scale=1.0)
        gpr=gp.GaussianProcessRegressor(kernel=kernel, random_state=0, n_restarts_optimizer=10)
        
        gpr.fit(X_trainGP,y_trainGP)
        y_pred,sigma = gpr.predict(X_train, return_std=True)
        
        r2 = metrics.r2_score(y_train, y_pred)
        MSE = metrics.mean_squared_error(y_train, y_pred)
        MAE = metrics.mean_absolute_error(y_train, y_pred)

    else:
        y_trainGP = dataL[:,-1]
        X_trainGP = dataL[:,0:-1]
        kernel = Matern(length_scale=1.0)
        gpr=gp.GaussianProcessRegressor(kernel=kernel, random_state=0, n_restarts_optimizer=10)
        
        gpr.fit(X_trainGP,y_trainGP)
        y_pred,sigma = gpr.predict(X_train, return_std=True)
        
        r2 = metrics.r2_score(y_train, y_pred)
        MSE = metrics.mean_squared_error(y_train, y_pred)
        MAE = metrics.mean_absolute_error(y_train, y_pred)
    
    return np.array([[r2]]), gpr, np.array([[MSE]]),np.array([[MAE]])


def computeR2_train_self(dataL, fs=False):
    '''
    Calculate the model performance for the labled data pool
    '''
    if fs:
        y_trainGP = dataL.iloc[:,-1]
        X_trainGP = dataL.iloc[:,0:-1]
        kernel = Matern(length_scale=1.0)
        gpr=gp.GaussianProcessRegressor(kernel=kernel, random_state=0, n_restarts_optimizer=10)
        
        gpr.fit(X_trainGP,y_trainGP)
        y_pred,sigma = gpr.predict(X_trainGP, return_std=True)
        
        r2 = metrics.r2_score(y_trainGP, y_pred)
        MSE = metrics.mean_squared_error(y_trainGP, y_pred)
        MAE = metrics.mean_absolute_error(y_trainGP, y_pred)

    else:
        y_trainGP = dataL[:,-1]
        X_trainGP = dataL[:,0:-1]
        kernel = Matern(length_scale=1.0)
        gpr=gp.GaussianProcessRegressor(kernel=kernel, random_state=0, n_restarts_optimizer=10)
        
        gpr.fit(X_trainGP,y_trainGP)
        y_pred,sigma = gpr.predict(X_trainGP, return_std=True)
        
        r2 = metrics.r2_score(y_trainGP, y_pred)
        MSE = metrics.mean_squared_error(y_trainGP, y_pred)
        MAE = metrics.mean_absolute_error(y_trainGP, y_pred)
  
    return np.array([[r2]]), gpr, np.array([[MSE]]),np.array([[MAE]])


def computeR2_unlabel(data, data_label, model, fs=False):
    '''
    Calculate the model performance for the unlabled data pool
    '''
    if fs:
        y_trainGP = data.iloc[:,-1]
        X_trainGP = data.iloc[:,0:-1]
        y_pred,sigma = model.predict(X_trainGP, return_std=True)
        sigma_average = np.mean(sigma)
    else: 
        y_trainGP = data[:,-1]
        X_trainGP = data[:,0:-1]
        y_pred,sigma = model.predict(X_trainGP, return_std=True)
        sigma_average = np.mean(sigma)

    return np.array([[sigma_average]])


def getBatch(dataPool, batchSz, fs=False):
    '''
    Get certain batch size of data

    Parameters:
        dataPool (numpy.ndarray): The unlabled data pool to select samples.
        batchSz (int): the batch size of selecting samples.

    Returns:
        if fs:
            dataBatch: the selected batch of sample from unlabeled data pool.
            dataPool: the updated unlabeled data pool.
            dataPool_ori: the unupdated data pool.
        else:
            dataBatch: the selected batch of sample from unlabeled data pool.
            dataPool: the updated unlabeled data pool.
    '''

    if fs:
        SelectIdx=np.random.choice(dataPool.index, batchSz, replace=False)
        # print(f'SelectIdx: {SelectIdx}')
        # remember the index not reset, so need to access the index by using loc
        dataBatch = dataPool.loc[SelectIdx]
        # dataBatch = dataPool.iloc[SelectIdx, :]
        dataPool = dataPool.drop(SelectIdx)
        # dataPool = dataPool.reset_index(drop=True)

    else: 
        SelectIdx=np.random.choice(dataPool.shape[0], batchSz, replace=False)
        dataBatch = dataPool[SelectIdx, :]
        dataPool = np.delete(dataPool,SelectIdx,0)

    return dataBatch, dataPool, SelectIdx


def getUcertainPoint(dataPool, cModel, fs=False): 
    '''
    Get the most uncertain sample from unlabeled data pool.

    Parameters:
        dataPool (numpy.ndarray): The unlabeled data pool to select samples.
        cModel (object): the model established based on labeled data pool.

    Returns:
        bestUcertainPoint: the selected most uncertain sample from unlabeled data pool.
        dataPool: the updated unlabeled data pool.
    '''
    if fs:
        y_pred,sigma = cModel.predict(dataPool.iloc[:,0:-1], return_std=True)
        ibest = sigma.argsort()[-1:][::-1]
        bestUcertainPoint = dataPool.loc[ibest,:]
        dataPool = dataPool.drop(ibest)
    else:
        y_pred,sigma = cModel.predict(dataPool[:,0:5], return_std=True)
        ibest = sigma.argsort()[-1:][::-1]
        bestUcertainPoint = dataPool[ibest,:]
        dataPool = np.delete(dataPool,ibest,0)

    return bestUcertainPoint, dataPool, ibest

def feature_selection(X, y, fs_score, iter, Alg, dname=True):
    '''
    Using feature selection method to get features index
    '''
    if dname:
        # Dictionary to map old column names to new names
        new_names = {
                    'ABS_f_D': 'd1',
                    'DISS_wf10_D': 'd2',
                    'STAT_e': 'd3',
                    'STAT_n_D': 'd4',
                    'STAT_n_A': 'd5',
                    'STAT_CC_D': 'd6',
                    'STAT_CC_A': 'd7',
                    'STAT_CC_D_An': 'd8',
                    'STAT_CC_A_Ca': 'd9',
                    'ABS_wf_D': 'd10',
                    'DISS_f10_D': 'd11',
                    'CT_f_e_conn': 'd12',
                    'CT_f_conn_D_An': 'd13',
                    'CT_f_conn_A_Ca': 'd14',
                    'CT_e_conn': 'd15',
                    'CT_e_D_An': 'd16',
                    'CT_e_A_Ca': 'd17',
                    'CT_f_D_tort1': 'd18',
                    'CT_f_A_tort1': 'd19',
                    'CT_n_D_adj_An': 'd20',
                    'CT_n_A_adj_Ca': 'd21',           
                }
        # Rename columns in the DataFrame
        df = X.rename(columns=new_names)
        feat_names=df.columns[:]

    else:
        feat_names = X.columns[:]

    rf = RandomForestRegressor(random_state=99)
    rf.fit(X, y)
    print("Features sorted by their score:")
    print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feat_names), reverse=True))

    plt.figure(figsize=(12,10))
    # plt.figure()
    importances=rf.feature_importances_
    x_plot = [2*i for i in range(len(importances))]
    indices=np.argsort(importances)
    # plt.title("Feature importances",fontsize=25)
    plt.barh(x_plot, importances[indices],height=1.8,color='#1f77b4')
    plt.xlabel("Feature importance score",fontsize=25)
    plt.ylabel("Features",fontsize=25)
    plt.yticks(x_plot, feat_names[indices],fontsize=20)
    plt.xticks(fontsize=20)
    plt.savefig('Results_Plot/'+f'{Alg}_results_iteration_{iter}.png', bbox_inches='tight')
    plt.close()
    # plt.show()

    final_indices = get_final_indices(importances, fs_score)

    return final_indices

def get_final_indices(importances_score, fs_score):
    '''
    Based on the requirement for feature importance score, choose the most important features'''
    
    indices = np.argsort(importances_score)[::-1]  # Sort indices in descending order of importances
    cumulative_sum = 0.0
    selected_indices = []
    
    for index in indices:
        cumulative_sum += importances_score[index]
        selected_indices.append(index)
        if cumulative_sum > fs_score:
            break
            
    return selected_indices