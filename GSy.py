from utils import computeR2
from utils import computeR2_train
from utils import computeR2_train_self
from utils import computeR2_unlabel
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def GSy_alg(X, y, labeledPoolN, runs=20):
    R2Smooth = []
    MSEsmooth = []
    MAEsmooth = []
    Infosmooth = []
    R2_train = []
    R2_trainS = []
    SelectData = []

    for rt in tqdm(range(runs)):
        np.random.seed(rt)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=rt)#test_size=0.2 for consistency
        dataPool = np.hstack((X_train, np.atleast_2d(y_train)))
        SelectIdx=np.random.choice(dataPool.shape[0], labeledPoolN, replace=False)
        dataPoolL = dataPool[SelectIdx, :]
        dataPool = np.delete(dataPool,SelectIdx,0)
        data = np.hstack((X_train, np.atleast_2d(y_train)))
        
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

        R2, Model, MSEstart, MAEstart = computeR2(dataPoolL, X_test, y_test)
        R2Res = np.append(R2Res, R2, axis=0)
        MSERes = np.append(MSERes, MSEstart, axis=0)
        MAERes = np.append(MAERes, MAEstart, axis=0)
        Info = computeR2_unlabel(dataPool, dataPoolL, Model)
        InfoRes = np.append(InfoRes, Info, axis=0)

        R2_t, Model, MSEstart_t,_ = computeR2_train(dataPoolL, X_train, y_train)
        R2Res_t = np.append(R2Res_t, R2_t, axis=0)

        R2_tS, ModelS, MSEstart_tS,_ = computeR2_train_self(dataPoolL)
        R2Res_tS = np.append(R2Res_tS, R2_tS, axis=0)

        for n in np.arange(10, 509):

            distY=np.zeros((dataPool[:,0:-1].shape[0],n))

            for i in np.arange(n):
                distY[:,i]= abs(Model.predict(dataPool[:,0:-1])-dataPoolL[i,-1]*np.ones((dataPool[:,0:-1].shape[0])))
    #         print(distY.shape)
            dist=distY.min(axis=1)

            idx2=np.argmax(dist)

            idsTest=np.delete(idsTest,idx2)
            databatch=dataPool[idx2,:]
            dataPool=data[idsTest,:]
            dataPoolL = np.vstack((dataPoolL, databatch))

            cR2, Model, cMSE, cMAE = computeR2(dataPoolL, X_test, y_test)
            R2Res = np.append(R2Res, cR2, axis=0)
            MSERes = np.append(MSERes, cMSE, axis=0)
            MAERes = np.append(MAERes, cMAE, axis=0)
            cInfo = computeR2_unlabel(dataPool, dataPoolL, Model)
            InfoRes = np.append(InfoRes, cInfo, axis=0)

            cR2_t, Model, cMSEstart_t,_ = computeR2_train(dataPoolL, X_train, y_train)
            R2Res_t = np.append(R2Res_t, cR2_t, axis=0)

            cR2_tS, ModelS, cMSEstart_tS,_ = computeR2_train_self(dataPoolL)
            R2Res_tS = np.append(R2Res_tS, cR2_tS, axis=0)

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