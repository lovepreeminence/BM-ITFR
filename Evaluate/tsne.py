import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def tsne_plot_embedding(data, label, filename, dirname, t=True):
    print('>>>tsne plotting')
    if t:
        #plot_embedding(data, label)
        tsne = TSNE(n_components = 2, init='pca', random_state = 0)
        data = tsne.fit_transform(data)

    fig = plt.figure()
    ax  = plt.subplot(111)
    plt.scatter(data[:, 0], data[:, 1], 10, c=label, cmap=plt.cm.Spectral, alpha=0.5)
    
    if not os.path.exists('./{}'.format(dirname)):
         os.mkdir('./{}'.format(dirname))

    plt.savefig("./{}/t-SNE-{}".format(dirname, filename))
    plt.close()
    print('>>>done tsne plot {}'.format(filename))


