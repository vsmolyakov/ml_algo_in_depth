import numpy as np
import pandas as pd
from scipy import linalg

from datetime import datetime
import pytz

from sklearn.datasets import make_sparse_spd_matrix
from sklearn.covariance import GraphicalLassoCV, ledoit_wolf
from sklearn.preprocessing import StandardScaler
from sklearn import cluster, manifold

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import pandas_datareader.data as web

np.random.seed(42)

def main():
        
    #generate data (synthetic)
    #num_samples = 60
    #num_features = 20
    #prec = make_sparse_spd_matrix(num_features, alpha=0.95, smallest_coef=0.4, largest_coef=0.7)
    #cov = linalg.inv(prec)
    #X = np.random.multivariate_normal(np.zeros(num_features), cov, size=num_samples)
    #X = StandardScaler().fit_transform(X)    
   
    #generate data (actual)
    STOCKS = {
        'SPY': 'S&P500',
        'LQD': 'Bond_Corp',
        'TIP': 'Bond_Treas',
        'GLD': 'Gold',
        'MSFT': 'Microsoft',
        'XOM':  'Exxon',
        'AMZN': 'Amazon',
        'BAC':  'BofA',
        'NVS':  'Novartis'}
      
    symbols, names = np.array(list(STOCKS.items())).T
    
    #load data
    #year, month, day, hour, minute, second, microsecond
    start = datetime(2015, 1, 1, 0, 0, 0, 0, pytz.utc)
    end = datetime(2017, 1, 1, 0, 0, 0, 0, pytz.utc)    

    qopen, qclose = [], []
    data_close, data_open = pd.DataFrame(), pd.DataFrame()    
    for ticker in symbols:
        price = web.DataReader(ticker, 'stooq', start, end)
        qopen.append(price['Open'])
        qclose.append(price['Close'])

    data_open = pd.concat(qopen, axis=1)
    data_open.columns = symbols
    data_close = pd.concat(qclose, axis=1)
    data_close.columns = symbols
    
    #per day variation in price for each symbol
    variation = data_close - data_open
    variation = variation.dropna()                        
                
    X = variation.values
    X /= X.std(axis=0) #standardize to use correlations rather than covariance
                
    #estimate inverse covariance    
    graph = GraphicalLassoCV()
    graph.fit(X)
    
    gl_cov = graph.covariance_
    gl_prec = graph.precision_
    gl_alphas =graph.cv_alphas_
    gl_scores = graph.cv_results_['mean_test_score']

    plt.figure()        
    sns.heatmap(gl_prec, xticklabels=names, yticklabels=names)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)    
    plt.tight_layout()
    plt.show()
    
    plt.figure()    
    plt.plot(gl_alphas, gl_scores, marker='o', color='b', lw=2.0, label='GraphLassoCV')
    plt.title("Graph Lasso Alpha Selection")
    plt.xlabel("alpha")
    plt.ylabel("score")
    plt.legend()
    plt.show()
    
    #cluster using affinity propagation
    _, labels = cluster.affinity_propagation(gl_cov)
    num_labels = np.max(labels)
    
    for i in range(num_labels+1):
        print("Cluster %i: %s" %((i+1), ', '.join(names[labels==i])))
    
    #find a low dim embedding for visualization
    node_model = manifold.LocallyLinearEmbedding(n_components=2, n_neighbors=6, eigen_solver='dense')
    embedding = node_model.fit_transform(X.T).T
    
    #generate plots
    plt.figure()
    plt.clf()
    ax = plt.axes([0.,0.,1.,1.])
    plt.axis('off')
    
    partial_corr = gl_prec
    d = 1 / np.sqrt(np.diag(partial_corr))    
    non_zero = (np.abs(np.triu(partial_corr, k=1)) > 0.02)  #connectivity matrix
    
    #plot the nodes
    plt.scatter(embedding[0], embedding[1], s = 100*d**2, c = labels, cmap = plt.cm.Spectral)
    
    #plot the edges
    start_idx, end_idx = np.where(non_zero)
    segments = [[embedding[:,start], embedding[:,stop]] for start, stop in zip(start_idx, end_idx)]
    values = np.abs(partial_corr[non_zero])
    lc = LineCollection(segments, zorder=0, cmap=plt.cm.hot_r, norm=plt.Normalize(0,0.7*values.max()))
    lc.set_array(values)
    lc.set_linewidths(2*values)
    ax.add_collection(lc)
    
    #plot the labels
    for index, (name, label, (x,y)) in enumerate(zip(names, labels, embedding.T)):
        plt.text(x,y,name,size=12)
    
    plt.show()
    
if __name__ == "__main__":
    main()    
        