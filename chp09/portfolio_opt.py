
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KDTree
from pandas.tools.plotting import scatter_matrix
from scipy.spatial import ConvexHull

import pandas_datareader.data as web
from datetime import datetime
import pytz

STOCKS = ['SPY','LQD','TIP','GLD','MSFT']

np.random.seed(0)    

if __name__ == "__main__":

    plt.close("all")
    
    #load data
    #year, month, day, hour, minute, second, microsecond
    start = datetime(2012, 1, 1, 0, 0, 0, 0, pytz.utc)
    end = datetime(2017, 1, 1, 0, 0, 0, 0, pytz.utc)     
    
    data = pd.DataFrame()
    series = []
    for ticker in STOCKS:
        price = web.DataReader(ticker, 'google', start, end)
        series.append(price['Close'])

    data = pd.concat(series, axis=1)
    data.columns = STOCKS
    data = data.dropna()
    
    #plot data correlations
    scatter_matrix(data, alpha=0.2, diagonal='kde')
    plt.show()

    #get current portfolio
    cash = 10000
    num_assets = np.size(STOCKS)
    cur_value = (1e4-5e3)*np.random.rand(num_assets,1) + 5e3        
    tot_value = np.sum(cur_value)
    weights = cur_value.ravel()/float(tot_value)
    
    #compute portfolio risk
    Sigma = data.cov().values
    Corr = data.corr().values        
    volatility = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
    
    plt.figure()
    plt.title('Correlation Matrix')        
    plt.imshow(Corr, cmap='gray')
    plt.xticks(range(len(STOCKS)),data.columns)
    plt.yticks(range(len(STOCKS)),data.columns)    
    plt.colorbar()
    plt.show()
        
    #generate random portfolio weights
    num_trials = 1000
    W = np.random.rand(num_trials, np.size(weights))    
    W = W/np.sum(W,axis=1).reshape(num_trials,1)  #normalize
    
    pv = np.zeros(num_trials)   #portoflio value  w'v
    ps = np.zeros(num_trials)   #portfolio sigma: sqrt(w'Sw)
    
    avg_price = data.mean().values
    adj_price = avg_price
    
    for i in range(num_trials):
        pv[i] = np.sum(adj_price * W[i,:])
        ps[i] = np.sqrt(np.dot(W[i,:].T, np.dot(Sigma, W[i,:])))
    
    points = np.vstack((ps,pv)).T
    hull = ConvexHull(points)
            
    plt.figure()
    plt.scatter(ps, pv, marker='o', color='b', linewidth = '3.0', label = 'tangent portfolio')
    plt.scatter(volatility, np.sum(adj_price * weights), marker = 's', color = 'r', linewidth = '3.0', label = 'current')
    plt.plot(points[hull.vertices,0], points[hull.vertices,1], linewidth = '2.0')    
    plt.title('expected return vs volatility')
    plt.ylabel('expected price')
    plt.xlabel('portfolio std dev')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    #query for nearest neighbor portfolio
    knn = 5    
    kdt = KDTree(points)
    query_point = np.array([2, 115]).reshape(1,-1)
    kdt_dist, kdt_idx = kdt.query(query_point,k=knn)
    print "top-%d closest to query portfolios:" %knn
    print "values: ", pv[kdt_idx.ravel()]
    print "sigmas: ", ps[kdt_idx.ravel()]
    
