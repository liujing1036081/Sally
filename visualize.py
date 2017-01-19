import matplotlib.pyplot as plt
import numpy as np
from statistics import mean


def draw_scatter(x, y, x_labels, y_labels, title=''):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x, y, marker='o', color='#78A5A3')
    plt.axhline(mean(y), color='#CE5A57')
    plt.axvline(mean(x), color='#CE5A57')
    plt.xlabel(x_labels)
    plt.ylabel(y_labels)
    plt.title(title)
    plt.show()




 if __name__ == '__main__':
     from load_data import load_CVAT_2

     # texts, valence, arousal = load_CVAT_2('./resources/CVAT2.0(sigma=1.0).csv')
     texts, valence, arousal = load_CVAT_2('./resources/corpus 2009 sigma 1.5.csv')
     draw_scatter(valence, arousal, 'Valence', 'Arousal')
