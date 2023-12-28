from utils import computeR2
from utils import computeR2_train
from utils import computeR2_train_self
from utils import computeR2_unlabel
from utils import feature_selection
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist, squareform
import pandas as pd

def GSy_alg(X, y, labeledPoolN, runs=20, freq=10, fs_score=0.98, Alg='GSy_fs'):
    R2Smooth = []
    MSEsmooth = []
    MAEsmooth = []
    Infosmooth = []
    R2_train = []
    R2_trainS = []
    SelectData = []

    for rt in tqdm(range(runs)):
        np.random.seed(rt)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=rt)
        dataPool = pd.concat([X_train, y_train], axis=1)
        SelectIdx=np.random.choice(dataPool.index, labeledPoolN, replace=False)
        dataPoolL = dataPool.iloc[SelectIdx, :]
        dataPool = dataPool.drop(SelectIdx)
        data = pd.concat([X_train, y_train], axis=1)
        
        Idx = []
        Idx = SelectIdx.tolist()
        #Idx.append(SelectIdx)
        idsTest=np.arange(0,len(y_train))
        idsTest=np.delete(idsTest,Idx)

        R2Res = np.empty((0,1), float)
        MSERes = np.empty((0,1), float)
        MAERes = np.empty((0,1), float)
        InfoRes = np.empty((0,1), float)
        R2Res_t = np.empty((0,1), float)
        R2Res_tS = np.empty((0,1), float)

        R2, Model, MSEstart, MAEstart = computeR2(dataPoolL, X_test, y_test, fs=True)
        R2Res = np.append(R2Res, R2, axis=0)
        MSERes = np.append(MSERes, MSEstart, axis=0)
        MAERes = np.append(MAERes, MAEstart, axis=0)
        Info = computeR2_unlabel(dataPool, dataPoolL, Model, fs=True)
        InfoRes = np.append(InfoRes, Info, axis=0)

        R2_t, Model, MSEstart_t,_ = computeR2_train(dataPoolL, X_train, y_train, fs=True)
        R2Res_t = np.append(R2Res_t, R2_t, axis=0)

        R2_tS, ModelS, MSEstart_tS,_ = computeR2_train_self(dataPoolL, fs=True)
        R2Res_tS = np.append(R2Res_tS, R2_tS, axis=0)

        # feature selection
        indices = feature_selection(dataPoolL.iloc[:, 0:-1],dataPoolL.iloc[:, -1], fs_score, 0, Alg)
        dataPoolL_fs = pd.concat([dataPoolL.iloc[:, 0:-1].iloc[:, indices],dataPoolL.iloc[:, -1]],axis=1)
        dataPool_fs = pd.concat([dataPool.iloc[:, 0:-1].iloc[:, indices], dataPool.iloc[:, -1]],axis=1)
        data_fs = pd.concat([data.iloc[:, 0:-1].iloc[:, indices], data.iloc[:, -1]],axis=1)

        # get model with fewer features
        _, Model_fs, _, _ = computeR2(dataPoolL_fs, X_test.iloc[:, indices], y_test, fs=True)

        for n in np.arange(10, 509):

            distY=np.zeros((dataPool_fs.iloc[:,0:-1].shape[0],n))

            for i in np.arange(n):
                distY[:,i]= abs(Model_fs.predict(dataPool_fs.iloc[:,0:-1])-dataPoolL_fs.iloc[i,-1]*np.ones((dataPool_fs.iloc[:,0:-1].shape[0])))
    #         print(distY.shape)
            dist=distY.min(axis=1)

            idx2=np.argmax(dist)
            idsTest=np.delete(idsTest,idx2)

            databatch_fs=dataPool_fs.iloc[idx2,:].to_frame().T
            dataPool_fs=data_fs.iloc[idsTest,:]
            dataPoolL_fs = pd.concat([dataPoolL_fs, databatch_fs], axis=0)
            
            databatch=dataPool.iloc[idx2,:].to_frame().T
            dataPool=data.iloc[idsTest,:]
            dataPoolL = pd.concat([dataPoolL, databatch], axis=0)

            cR2, Model, cMSE, cMAE = computeR2(dataPoolL, X_test, y_test, fs=True)
            R2Res = np.append(R2Res, cR2, axis=0)
            MSERes = np.append(MSERes, cMSE, axis=0)
            MAERes = np.append(MAERes, cMAE, axis=0)
            cInfo = computeR2_unlabel(dataPool, dataPoolL, Model, fs=True)
            InfoRes = np.append(InfoRes, cInfo, axis=0)

            cR2_t, Model, cMSEstart_t,_ = computeR2_train(dataPoolL, X_train, y_train, fs=True)
            R2Res_t = np.append(R2Res_t, cR2_t, axis=0)

            cR2_tS, ModelS, cMSEstart_tS,_ = computeR2_train_self(dataPoolL, fs=True)
            R2Res_tS = np.append(R2Res_tS, cR2_tS, axis=0)

            if i % freq == 0:

                # feature selection
                indices = feature_selection(dataPoolL.iloc[:, 0:-1],dataPoolL.iloc[:, -1], fs_score, 0, Alg)
                dataPoolL_fs = pd.concat([dataPoolL.iloc[:, 0:-1].iloc[:, indices],dataPoolL.iloc[:, -1]],axis=1)
                dataPool_fs = pd.concat([dataPool.iloc[:, 0:-1].iloc[:, indices], dataPool.iloc[:, -1]],axis=1)
                data_fs = pd.concat([data.iloc[:, 0:-1].iloc[:, indices], data.iloc[:, -1]],axis=1)

                # get model with fewer features
                _, Model_fs, _,_ = computeR2(dataPoolL_fs, X_test.iloc[:, indices], y_test, fs=True)

        R2Smooth.append(R2Res)
        MSEsmooth.append(MSERes)
        MAEsmooth.append(MAERes)
        Infosmooth.append(InfoRes)
        R2_train.append(R2Res_t)
        R2_trainS.append(R2Res_tS)
        SelectData.append(dataPoolL)
        
    #     SelectData.append(dataPoolL)
    R2Smooth = np.asarray(R2Smooth)
    MSEsmooth = np.asarray(MSEsmooth)
    MAEsmooth = np.asarray(MAEsmooth)
    InfoSmooth = np.asarray(Infosmooth)
    R2_train = np.asarray(R2_train) 
    R2_trainS = np.asarray(R2_trainS) 

    R2Smooth_std = np.std(R2Smooth, axis=0)#note variable update
    accuracySmooth = np.mean(R2Smooth, axis=0)

    MSEsmooth_std = np.std(MSEsmooth, axis=0)
    MSEsmooth = np.mean(MSEsmooth, axis=0)
    
    MAEsmooth_std = np.std(MAEsmooth, axis=0)
    MAEsmooth = np.mean(MAEsmooth, axis=0)
    
    InfoSmooth_std = np.std(InfoSmooth, axis=0)
    InfoSmooth_mean = np.mean(InfoSmooth, axis=0)

    R2_train_std = np.std(R2_train, axis=0)
    R2_train_mean = np.mean(R2_train, axis=0)

    R2_train_stdS = np.std(R2_trainS, axis=0)
    R2_train_meanS = np.mean(R2_trainS, axis=0)

    return (R2Smooth_std, accuracySmooth, InfoSmooth_std, InfoSmooth_mean,\
            MSEsmooth_std,MSEsmooth,MAEsmooth_std, MAEsmooth,\
            R2_train_std, R2_train_mean, R2_train_stdS, R2_train_meanS, SelectData)