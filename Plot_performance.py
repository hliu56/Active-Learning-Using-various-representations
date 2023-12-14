import matplotlib.pyplot as plt
import numpy as np


def plot_performance(value, variance, Alg):

    x_values = np.arange(len(value))

    plt.plot(x_values, value, label=Alg)
    plt.fill_between(x_values, (value+variance).flatten(), (value-variance).flatten(), alpha=0.2)
    plt.xlabel('Number of samples')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig('Results_Plot/'+f'PerformancePlot_{Alg}.png')
