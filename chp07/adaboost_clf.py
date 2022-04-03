import itertools
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn import datasets

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score, train_test_split

from mlxtend.plotting import plot_learning_curves
from mlxtend.plotting import plot_decision_regions

def main():

    iris = datasets.load_iris()
    X, y = iris.data[:, 0:2], iris.target
    
    #XOR dataset
    #X = np.random.randn(200, 2)
    #y = np.array(map(int,np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)))
    
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=1)

    num_est = [1, 2, 3, 10]
    label = ['AdaBoost (n_est=1)', 'AdaBoost (n_est=2)', 'AdaBoost (n_est=3)', 'AdaBoost (n_est=10)']
    
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2)
    grid = itertools.product([0,1],repeat=2)

    for n_est, label, grd in zip(num_est, label, grid):     
        boosting = AdaBoostClassifier(base_estimator=clf, n_estimators=n_est)   
        boosting.fit(X, y)
        ax = plt.subplot(gs[grd[0], grd[1]])
        fig = plot_decision_regions(X=X, y=y, clf=boosting, legend=2)
        plt.title(label)

    plt.show()
    #plt.savefig('./figures/boosting_ensemble.png')             

    #plot learning curves
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    boosting = AdaBoostClassifier(base_estimator=clf, n_estimators=10)
        
    plt.figure()
    plot_learning_curves(X_train, y_train, X_test, y_test, boosting, print_model=False, style='ggplot')
    plt.show()
    #plt.savefig('./figures/boosting_ensemble_learning_curve.png')             

    #Ensemble Size
    num_est = list(map(int, np.linspace(1,100,20)))
    bg_clf_cv_mean = []
    bg_clf_cv_std = []
    for n_est in num_est:
        print("num_est: ", n_est)
        ada_clf = AdaBoostClassifier(base_estimator=clf, n_estimators=n_est)
        scores = cross_val_score(ada_clf, X, y, cv=3, scoring='accuracy')
        bg_clf_cv_mean.append(scores.mean())
        bg_clf_cv_std.append(scores.std())
        
    plt.figure()
    (_, caps, _) = plt.errorbar(num_est, bg_clf_cv_mean, yerr=bg_clf_cv_std, c='blue', fmt='-o', capsize=5)
    for cap in caps:
        cap.set_markeredgewidth(1)                                                                                                                                
    plt.ylabel('Accuracy'); plt.xlabel('Ensemble Size'); plt.title('AdaBoost Ensemble');
    plt.show()
    #plt.savefig('./figures/boosting_ensemble_size.png')

if __name__ == "__main__":    
    main()