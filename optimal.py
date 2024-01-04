from utils import computeR2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def Optimal_test(X, y, runs=20):
    R2Smooth = []
    MSEsmooth = []
    MAEsmooth = []
    SelectData = []
        
    for rt in tqdm(range(runs)):
        np.random.seed(rt) # set the same random choice for initial codition
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rt)
        dataPoolL = np.hstack((X_train, np.atleast_2d(y_train)))
        R2, Model, MSEstart, MAEstart = computeR2(dataPoolL, X_test, y_test)

        R2Res = np.empty((0,1), float)
        MSERes = np.empty((0,1), float)
        MAERes = np.empty((0,1), float)

        R2Res = np.append(R2Res, R2, axis=0)
        MSERes = np.append(MSERes, MSEstart, axis=0)
        MAERes = np.append(MAERes, MAEstart, axis=0)
        
        R2Smooth.append(R2Res)
        MSEsmooth.append(MSERes)
        MAEsmooth.append(MAERes)
        SelectData.append(dataPoolL)
    
    R2Smooth = np.asarray(R2Smooth)
    MSEsmooth = np.asarray(MSEsmooth)
    MAEsmooth = np.asarray(MAEsmooth)
    # R2_train = np.asarray(R2_train) 
    # R2_trainS = np.asarray(R2_trainS)

    R2Smooth_std = np.std(R2Smooth, axis=0)#note variable update
    accuracySmooth = np.mean(R2Smooth, axis=0)

    MSEsmooth_std = np.std(MSEsmooth, axis=0)
    MSEsmooth = np.mean(MSEsmooth, axis=0)
    
    MAEsmooth_std = np.std(MAEsmooth, axis=0)
    MAEsmooth = np.mean(MAEsmooth, axis=0)

    return (R2Smooth_std, accuracySmooth, \
            MSEsmooth_std,MSEsmooth,MAEsmooth_std, MAEsmooth,\
            SelectData)