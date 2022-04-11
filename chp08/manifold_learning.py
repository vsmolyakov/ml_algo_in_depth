import numpy as np
import matplotlib.pyplot as plt

from time import time
from sklearn import manifold

from sklearn.datasets import load_digits
from sklearn.neighbors import KDTree

def plot_digits(X):

    n_img_per_row = np.amin((20, np.int(np.sqrt(X.shape[0]))))
    img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
    for i in range(n_img_per_row):
        ix = 10 * i + 1
        for j in range(n_img_per_row):
            iy = 10 * j + 1
            img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))

    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.title('A selection from the 64-dimensional digits dataset')    

def mnist_manifold():
            
    digits = load_digits()
    
    X = digits.data
    y = digits.target
    
    num_classes = np.unique(y).shape[0]
        
    plot_digits(X)
        
    #TSNE 
    #Barnes-Hut: O(d NlogN) where d is dim and N is the number of samples
    #Exact: O(d N^2)
    t0 = time()
    tsne = manifold.TSNE(n_components = 2, init = 'pca', method = 'barnes_hut', verbose = 1)
    X_tsne = tsne.fit_transform(X)
    t1 = time()
    print('t-SNE: %.2f sec' %(t1-t0))
    tsne.get_params()
    
    plt.figure()
    for k in range(num_classes):
        plt.plot(X_tsne[y==k,0], X_tsne[y==k,1],'o')
    plt.title('t-SNE embedding of digits dataset')
    plt.xlabel('X1')
    plt.ylabel('X2')
    axes = plt.gca()
    axes.set_xlim([X_tsne[:,0].min()-1,X_tsne[:,0].max()+1])
    axes.set_ylim([X_tsne[:,1].min()-1,X_tsne[:,1].max()+1])
    plt.show()
        
    #ISOMAP
    #1. Nearest neighbors search: O(d log k N log N)
    #2. Shortest path graph search: O(N^2(k+log(N))
    #3. Partial eigenvalue decomposition: O(dN^2)
    
    t0 = time()
    isomap = manifold.Isomap(n_neighbors = 5, n_components = 2)
    X_isomap = isomap.fit_transform(X)
    t1 = time()
    print('Isomap: %.2f sec' %(t1-t0))
    isomap.get_params()
    
    plt.figure()
    for k in range(num_classes):
        plt.plot(X_isomap[y==k,0], X_isomap[y==k,1], 'o', label=str(k), linewidth = 2)
    plt.title('Isomap embedding of the digits dataset')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    
    #Use KD-tree to find k-nearest neighbors to a query image
    kdt = KDTree(X_isomap)
    Q = np.array([[-160, -30],[-102, 14]])
    kdt_dist, kdt_idx = kdt.query(Q,k=20)
    plot_digits(X[kdt_idx.ravel(),:])
                                       
if __name__ == "__main__":    
    mnist_manifold()
        
    