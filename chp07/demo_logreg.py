import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from active_learning import ActiveLearner
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)

def main():
    
    #number of labeled points
    num_queries = 30
    
    #generate data
    data, target = make_classification(n_samples=200, n_features=2, n_informative=2,\
                                       n_redundant=0, n_classes=2, weights = [0.5, 0.5], random_state=0)
    
    #split into labeled and unlabeled pools
    X_train, X_unlabeled, y_train, y_oracle = train_test_split(data, target, test_size=0.2, random_state=0)        

    #random sub-sampling        
    rnd_idx = np.random.randint(0, X_train.shape[0], num_queries)    
    X1 = X_train[rnd_idx,:]
    y1 = y_train[rnd_idx]            
    
    clf1 = LogisticRegression()
    clf1.fit(X1, y1)
    
    y1_preds = clf1.predict(X_unlabeled)
    score1 = accuracy_score(y_oracle, y1_preds)
    print("random subsampling accuracy: ", score1)
    
    #plot 2D decision boundary: w2x2 + w1x1 + w0 = 0
    w0 = clf1.intercept_
    w1, w2 = clf1.coef_[0]
    xx = np.linspace(-1, 1, 100)
    decision_boundary = -w0/float(w2) - (w1/float(w2))*xx
    
    plt.figure()
    plt.scatter(data[rnd_idx,0], data[rnd_idx,1], c='black', marker='s', s=64, label='labeled')    
    plt.scatter(data[target==0,0], data[target==0,1], c='blue', marker='o', alpha=0.5, label='class 0')
    plt.scatter(data[target==1,0], data[target==1,1], c='red', marker='o', alpha=0.5, label='class 1')
    plt.plot(xx, decision_boundary, linewidth = 2.0, c='black', linestyle = '--', label='log reg boundary')
    plt.title("Random Subsampling")
    plt.legend()
    plt.show()
    
    #active learning        
    AL = ActiveLearner(strategy='entropy')
    al_idx = AL.rank(clf1, X_unlabeled, num_queries=num_queries)    
    
    X2 = X_train[al_idx,:]
    y2 = y_train[al_idx]
    
    clf2 = LogisticRegression()
    clf2.fit(X2, y2)

    y2_preds = clf2.predict(X_unlabeled)
    score2 = accuracy_score(y_oracle, y2_preds)
    print("active learning accuracy: ", score2)
                            
    #plot 2D decision boundary: w2x2 + w1x1 + w0 = 0
    w0 = clf2.intercept_
    w1, w2 = clf2.coef_[0]        
    xx = np.linspace(-1, 1, 100)
    decision_boundary = -w0/float(w2) - (w1/float(w2))*xx
    
    plt.figure()
    plt.scatter(data[al_idx,0], data[al_idx,1], c='black', marker='s', s=64, label='labeled')        
    plt.scatter(data[target==0,0], data[target==0,1], c='blue', marker='o', alpha=0.5, label='class 0')
    plt.scatter(data[target==1,0], data[target==1,1], c='red', marker='o', alpha=0.5, label='class 1')
    plt.plot(xx, decision_boundary, linewidth = 2.0, c='black', linestyle = '--', label='log reg boundary')
    plt.title("Uncertainty Sampling")
    plt.legend()
    plt.show()
            
if __name__ == "__main__":
    
    main()