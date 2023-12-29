import os
import pickle
from datetime import datetime
current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

def save_data(r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, Alg):

    results = {
    'R2Smooth_std1': r1,
    'accuracySmooth1': r2,
    'InfoSmooth_std1': r3,
    'InfoSmooth_mean1': r4,
    'MSEsmooth_std1': r5,
    'MSEsmooth1': r6,
    'MAEsmooth_std1': r7,
    'MAEsmooth1': r8,
    'R2_train_std1': r9,
    'R2_train_mean1': r10,
    'R2_train_stdS1': r11,
    'R2_train_meanS1': r12,
    'SelectData1': r13}

    # Define the path to your local folder and filename
    folder_path = 'Results_Data/'
    file_name = f'{Alg}_results_data_{formatted_datetime}.pkl'

    # Save the results to a file in the specified folder
    with open(folder_path + file_name, 'wb') as file:
        pickle.dump(results, file)

    


