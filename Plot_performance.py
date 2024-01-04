import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

def plot_performance(value, variance, Alg, errorbar=False):
    
    if errorbar:
        fig, ax = plt.subplots()

        offset = 10
        error_freq = 30
        y=value
        e=variance
        x = np.arange(len(y))
        p = ax.plot(x, y, label=f'{Alg}', lw=3)

        xe, ye, ee = x[offset::error_freq], y[offset::error_freq], e[offset::error_freq]
        xe, ye, ee = xe.flatten(), ye.flatten(), ee.flatten()
        ax.errorbar(xe, ye, yerr=ee, alpha=0.3, ls='none', ecolor=p[0].get_color(), elinewidth=3, capsize=4, capthick=3)
        offset += error_freq
        # ax.set_xlim(0, len(MAEsmooth1[0]))  # Assuming all arrays have the same length
        # ax.set_ylim(0, max(max(MAEsmooth1), max(MAEsmooth_std1)) * 1.1)

        plt.xlabel('Number of samples')
        plt.ylabel('MAE')
        plt.title(f'MAE Plot for {Alg}')
        plt.legend()
        # plt.grid(True)
        plt.savefig('Results_Plot_errorbar/'+f'PerformancePlot_{Alg}_{formatted_datetime}.png')
        print('Plot completed')

    else:
        x_values = np.arange(len(value))

        plt.plot(x_values, value, label=Alg)
        plt.fill_between(x_values, (value+variance).flatten(), (value-variance).flatten(), alpha=0.2)
        plt.xlabel('Number of samples')
        plt.ylabel('MAE')
        plt.legend()
        plt.savefig('Results_Plot/'+f'PerformancePlot_{Alg}_{formatted_datetime}.png')
        print('Plot completed')
