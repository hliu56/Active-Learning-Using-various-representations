import iGS
import GSx
import GSy
import Uncertainty
from Random import RandomSampling
from utils import get_input
from utils import normalized_data
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Hyperparameters
numDataTotal = 509
labeledPoolN = 10
batchSz = 1
nAccs = (numDataTotal-labeledPoolN)//batchSz
RepeatTimes = 1
# Get the input files
myfile =r'Data/CombinedPSP.csv'
X, y = get_input(myfile)

def main(save=False):
    # Run different algorithms to get model performance and labeled data pool
    R2Smooth_std1, accuracySmooth1, InfoSmooth_std1, InfoSmooth_mean1,\
    MSEsmooth_std1,MSEsmooth1,MAEsmooth_std1, MAEsmooth1,\
    R2_train_std1, R2_train_mean1, R2_train_stdS1, R2_train_meanS1, SelectData1 = RandomSampling(X, y, labeledPoolN, runs=RepeatTimes)
    
    # R2Smooth_std2, accuracySmooth2, InfoSmooth_std2, InfoSmooth_mean2,\
    # MSEsmooth_std2,MSEsmooth2,MAEsmooth_std2, MAEsmooth2,\
    # R2_train_std2, R2_train_mean2, R2_train_stdS2, R2_train_meanS2, SelectData2 = Uncertainty(X, y, runs=RepeatTimes )
    
    # R2Smooth_std3, accuracySmooth3, InfoSmooth_std3, InfoSmooth_mean3,\
    # MSEsmooth_std3,MSEsmooth3,MAEsmooth_std3, MAEsmooth3,\
    # R2_train_std3, R2_train_mean3, R2_train_stdS3, R2_train_meanS3, SelectData3 = GSx(X, y, runs=RepeatTimes)

    # R2Smooth_std4, accuracySmooth4, InfoSmooth_std4, InfoSmooth_mean4,\
    # MSEsmooth_std4,MSEsmooth4,MAEsmooth_std4, MAEsmooth4,\
    # R2_train_std4, R2_train_mean4, R2_train_stdS4, R2_train_meanS4, SelectData4 = GSy(X, y, runs=RepeatTimes)
    
    # R2Smooth_std5, accuracySmooth5, InfoSmooth_std5, InfoSmooth_mean5,\
    # MSEsmooth_std5,MSEsmooth5,MAEsmooth_std5, MAEsmooth5,\
    # R2_train_std5, R2_train_mean5, R2_train_stdS5, R2_train_meanS5, SelectData5 = iGS(X, y, runs=RepeatTimes)

    # Save data
    if save:

    # Plotting
    ...
    


if __name__=="__main__":
    main()