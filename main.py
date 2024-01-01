from Random import RandomSampling
from Uncertainty import UncertaintySampling
from GSx import GSx_alg
from GSy import GSy_alg
from iGS import iGS_alg

from Random_fs import RandomSampling_fs
from Uncertainty_fs import UncertaintySampling_fs
from GSx_fs import GSx_alg_fs


from utils import get_input
from utils import get_input_all
from utils import normalized_data
from Save_data import save_data
from Plot_performance import plot_performance
import pickle
import numpy as np
import matplotlib.pyplot as plt
import datetime
import warnings
warnings.filterwarnings('ignore')

# Hyperparameters
numDataTotal = 509
labeledPoolN = 10
batchSz = 1
nAccs = (numDataTotal-labeledPoolN)//batchSz
RepeatTimes = 20
# Get the input files
myfile =r'Data/CombinedPSP.csv'
# X, y = get_input(myfile)
# If combine feature selection
# X, y = get_input_all(myfile)

def main(features_know=False):
    '''
    set the features_know = True if the features are known
    '''
    # Run different algorithms to get model performance and labeled data pool
    if features_know:
        X, y = get_input(myfile)
        # Alg='Random'
        # R2Smooth_std1, accuracySmooth1, InfoSmooth_std1, InfoSmooth_mean1,\
        # MSEsmooth_std1,MSEsmooth1,MAEsmooth_std1, MAEsmooth1,\
        # R2_train_std1, R2_train_mean1, R2_train_stdS1, R2_train_meanS1, SelectData1 = RandomSampling(X, y, labeledPoolN, runs=RepeatTimes)
        #12min36s/2run
        #2h19min33s/20run
        
        # Alg='Uncertainty'
        # R2Smooth_std1, accuracySmooth1, InfoSmooth_std1, InfoSmooth_mean1,\
        # MSEsmooth_std1,MSEsmooth1,MAEsmooth_std1, MAEsmooth1,\
        # R2_train_std1, R2_train_mean1, R2_train_stdS1, R2_train_meanS1, SelectData1 = UncertaintySampling(X, y, labeledPoolN, runs=RepeatTimes )
        # 2h02min33s/20runs
        
        # Alg='GSx'
        # R2Smooth_std1, accuracySmooth1, InfoSmooth_std1, InfoSmooth_mean1,\
        # MSEsmooth_std1,MSEsmooth1,MAEsmooth_std1, MAEsmooth1,\
        # R2_train_std1, R2_train_mean1, R2_train_stdS1, R2_train_meanS1, SelectData1 = GSx_alg(X, y, labeledPoolN, runs=RepeatTimes)
        #1h58min49s/20runs

        # Alg='GSy'
        # R2Smooth_std1, accuracySmooth1, InfoSmooth_std1, InfoSmooth_mean1,\
        # MSEsmooth_std1,MSEsmooth1,MAEsmooth_std1, MAEsmooth1,\
        # R2_train_std1, R2_train_mean1, R2_train_stdS1, R2_train_meanS1, SelectData1 = GSy_alg(X, y, labeledPoolN, runs=RepeatTimes)
        # 5h19min02s/20runs
        
        Alg='iGS'
        R2Smooth_std1, accuracySmooth1, InfoSmooth_std1, InfoSmooth_mean1,\
        MSEsmooth_std1,MSEsmooth1,MAEsmooth_std1, MAEsmooth1,\
        R2_train_std1, R2_train_mean1, R2_train_stdS1, R2_train_meanS1, SelectData1 = iGS_alg(X, y, labeledPoolN, runs=RepeatTimes)
        # 5:27:02/20runs


    else:
        X, y = get_input_all(myfile)
        # AL&FS
        # Alg='Random_fs'
        # R2Smooth_std1, accuracySmooth1, InfoSmooth_std1, InfoSmooth_mean1,\
        # MSEsmooth_std1,MSEsmooth1,MAEsmooth_std1, MAEsmooth1,\
        # R2_train_std1, R2_train_mean1, R2_train_stdS1, R2_train_meanS1, SelectData1 = \
        #     RandomSampling_fs(X, y, labeledPoolN, runs=RepeatTimes, freq=10, fs_score=0.98, Alg=Alg)
        # 13min37s/2 run
        # 6:53:20

        # Alg='Uncertainty_fs'
        # R2Smooth_std1, accuracySmooth1, InfoSmooth_std1, InfoSmooth_mean1,\
        # MSEsmooth_std1,MSEsmooth1,MAEsmooth_std1, MAEsmooth1,\
        # R2_train_std1, R2_train_mean1, R2_train_stdS1, R2_train_meanS1, SelectData1 = \
        #     UncertaintySampling_fs(X, y, labeledPoolN, runs=RepeatTimes, freq=10, fs_score=0.98, Alg=Alg)
        # 1:33:56

        Alg='GSx_alg_fs'
        R2Smooth_std1, accuracySmooth1, InfoSmooth_std1, InfoSmooth_mean1,\
        MSEsmooth_std1,MSEsmooth1,MAEsmooth_std1, MAEsmooth1,\
        R2_train_std1, R2_train_mean1, R2_train_stdS1, R2_train_meanS1, SelectData1 = \
            GSx_alg_fs(X, y, labeledPoolN, runs=RepeatTimes, freq=10, fs_score=0.98, Alg=Alg)

    # Save data
    save_data(R2Smooth_std1, accuracySmooth1, InfoSmooth_std1, InfoSmooth_mean1,\
                        MSEsmooth_std1,MSEsmooth1,MAEsmooth_std1, MAEsmooth1,\
                        R2_train_std1, R2_train_mean1, R2_train_stdS1, R2_train_meanS1, SelectData1, Alg)
        
        # results3 = save_data(R2Smooth_std3, accuracySmooth3, InfoSmooth_std3, InfoSmooth_mean3,\
        #                     MSEsmooth_std3,MSEsmooth3,MAEsmooth_std3, MAEsmooth3,\
        #                     R2_train_std3, R2_train_mean3, R2_train_stdS3, R2_train_meanS3, SelectData3, Alg)
    


    # Plotting
    plot_performance(MAEsmooth1, MAEsmooth_std1, Alg)
    # plot_performance(MAEsmooth3, MAEsmooth_std3, Alg)
    


if __name__=="__main__":
    main()