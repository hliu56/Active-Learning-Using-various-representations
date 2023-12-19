from utils import computeR2
from utils import computeR2_train
from utils import computeR2_train_self
from utils import computeR2_unlabel
from utils import getBatch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def RandomSampling(X, y, labeledPoolN, runs=20):
    R2Smooth = []
    MSEsmooth = []
    MAEsmooth = []
    Infosmooth = []
    R2_train = []
    R2_trainS = []
    SelectData = []
        
    for rt in tqdm(range(runs)):
        np.random.seed(rt) # set the same random choice for initial codition
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rt)
        dataPool = np.hstack((X_train, np.atleast_2d(y_train)))
        SelectIdx=np.random.choice(dataPool.shape[0], labeledPoolN, replace=False)
        dataPoolL = dataPool[SelectIdx, :]
        dataPool = np.delete(dataPool,SelectIdx,0)

        R2Res = np.empty((0,1), float)
        MSERes = np.empty((0,1), float)
        MAERes = np.empty((0,1), float)
        InfoRes = np.empty((0,1), float)
        R2Res_t = np.empty((0,1), float)
        R2Res_tS = np.empty((0,1), float)

        R2, Model, MSEstart, MAEstart = computeR2(dataPoolL, X_test, y_test)
        R2_t, Model_t, MSEstart_t, _ = computeR2_train(dataPoolL, X_train, y_train)
        R2_tS, Model_tS, MSEstart_tS, _ = computeR2_train_self(dataPoolL)
        Info = computeR2_unlabel(dataPool, dataPoolL, Model)
        print(f'Info.shape: {Info.shape}')
        print(f'R2.shape {R2.shape}')

        R2Res = np.append(R2Res, R2, axis=0)
        MSERes = np.append(MSERes, MSEstart, axis=0)
        MAERes = np.append(MAERes, MAEstart, axis=0)
        InfoRes = np.append(InfoRes, Info, axis=0)
        print(f'InfoRes {InfoRes}')

        R2Res_t = np.append(R2Res_t, R2_t, axis=0)
        R2Res_tS = np.append(R2Res_tS, R2_tS, axis=0)

        for i in range(499):
            dataBatch, dataPool, _ = getBatch(dataPool, 1)
            dataPoolL = np.vstack((dataPoolL, dataBatch))

            cR2, Model, cMSE, cMAE = computeR2(dataPoolL, X_test, y_test)
            cR2_t, Model_t, cMSEstart_t,_ = computeR2_train(dataPoolL, X_train, y_train)
            cR2_tS, Model_tS, cMSEstart_tS,_ = computeR2_train_self(dataPoolL)
            cInfo = computeR2_unlabel(dataPool, dataPoolL, Model)

            R2Res = np.append(R2Res, cR2, axis=0)
            MSERes = np.append(MSERes, cMSE, axis=0)
            MAERes = np.append(MAERes, cMAE, axis=0)
            InfoRes = np.append(InfoRes, cInfo, axis=0)
            R2Res_t = np.append(R2Res_t, cR2_t, axis=0)
            R2Res_tS = np.append(R2Res_tS, cR2_tS, axis=0)
        
        R2Smooth.append(R2Res)
        MSEsmooth.append(MSERes)
        MAEsmooth.append(MAERes)
        Infosmooth.append(InfoRes)
        R2_train.append(R2Res_t)
        R2_trainS.append(R2Res_tS)
        SelectData.append(dataPoolL)
    
    R2Smooth = np.asarray(R2Smooth)
    MSEsmooth = np.asarray(MSEsmooth)
    MAEsmooth = np.asarray(MAEsmooth)
    InfoSmooth = np.asarray(Infosmooth)
    R2_train = np.asarray(R2_train) 
    R2_trainS = np.asarray(R2_trainS)

    R2Smooth_std = np.std(R2Smooth, axis=0)#note variable update
    accuracySmooth = np.mean(R2Smooth, axis=0)
    
    InfoSmooth_std = np.std(InfoSmooth, axis=0)
    InfoSmooth_mean = np.mean(InfoSmooth, axis=0)
    # for train set
    R2_train_std = np.std(R2_train, axis=0)
    R2_train_mean = np.mean(R2_train, axis=0)
    # FOR TRAIN ITSELF
    R2_train_stdS = np.std(R2_trainS, axis=0)
    R2_train_meanS = np.mean(R2_trainS, axis=0)

    MSEsmooth_std = np.std(MSEsmooth, axis=0)
    MSEsmooth = np.mean(MSEsmooth, axis=0)
    
    MAEsmooth_std = np.std(MAEsmooth, axis=0)
    MAEsmooth = np.mean(MAEsmooth, axis=0)

    return (R2Smooth_std, accuracySmooth, InfoSmooth_std, InfoSmooth_mean,\
            MSEsmooth_std,MSEsmooth,MAEsmooth_std, MAEsmooth,\
            R2_train_std, R2_train_mean, R2_train_stdS, R2_train_meanS, SelectData)