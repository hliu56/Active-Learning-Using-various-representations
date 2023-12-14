import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing
import sklearn.gaussian_process as gp
from sklearn.gaussian_process.kernels import Matern


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
def computeR2(dataL, X_test, y_test):
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

def computeR2_train(dataL, X_train, y_train):
    '''
    Calculate the model performance for the initial 80% training data set
    '''

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

def computeR2_train_self(dataL):
    '''
    Calculate the model performance for the labled data pool
    '''

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

def computeR2_unlabel(data, data_label, model):
    '''
    Calculate the model performance for the unlabled data pool
    '''
     
    y_trainGP = data[:,-1]
    X_trainGP = data[:,0:-1]
    y_pred,sigma = model.predict(X_trainGP, return_std=True)
    sigma_average = np.mean(sigma)
    return np.array([[sigma_average]])


def getBatch(dataPool, batchSz):
    '''
    Get certain batch size of data

    Parameters:
        dataPool (numpy.ndarray): The unlabled data pool to select samples.
        batchSz (int): the batch size of selecting samples.

    Returns:
        dataBatch: the selected batch of sample from unlabeled data pool.
        dataPool: the updated unlabeled data pool.
    '''
    
    SelectIdx=np.random.choice(dataPool.shape[0], batchSz, replace=False)
    dataBatch = dataPool[SelectIdx, :]
    dataPool = np.delete(dataPool,SelectIdx,0)
    
    return dataBatch, dataPool


def getUcertainPoint(dataPool, cModel): 
    '''
    Get the most uncertain sample from unlabeled data pool.

    Parameters:
        dataPool (numpy.ndarray): The unlabeled data pool to select samples.
        cModel (object): the model established based on labeled data pool.

    Returns:
        bestUcertainPoint: the selected most uncertain sample from unlabeled data pool.
        dataPool: the updated unlabeled data pool.
    '''

    y_pred,sigma = cModel.predict(dataPool[:,0:5], return_std=True)
    ibest = sigma.argsort()[-1:][::-1]
    bestUcertainPoint = dataPool[ibest,:]
    dataPool = np.delete(dataPool,ibest,0)
    return bestUcertainPoint, dataPool