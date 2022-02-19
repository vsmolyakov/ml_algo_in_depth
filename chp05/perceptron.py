import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 

from scipy.stats import randint
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

class perceptron:
    def __init__(self, num_epochs, dim):
        self.num_epochs = num_epochs
        self.theta0 = 0
        self.theta = np.zeros(dim)

    def fit(self, X_train, y_train):
        n = X_train.shape[0]
        dim = X_train.shape[1]

        k = 1
        for epoch in range(self.num_epochs):
            for i in range(n):
                #sample random point
                idx = randint.rvs(0, n-1, size=1)[0] 
                #hinge loss
                if (y_train[idx] * (np.dot(self.theta, X_train[idx,:]) + self.theta0) <= 0):
                    #update learning rate
                    eta = pow(k+1, -1)
                    k += 1
                    #print("eta: ", eta)

                    #update theta
                    self.theta = self.theta + eta * y_train[idx] * X_train[idx, :]
                    self.theta0 = self.theta0 + eta * y_train[idx]            
                #end if
            print("epoch: ", epoch)
            print("theta: ", self.theta)
            print("theta0: ", self.theta0)
            #end for
        #end for

    def predict(self, X_test):
        n = X_test.shape[0]
        dim = X_test.shape[1]

        y_pred = np.zeros(n)
        for idx in range(n):
            y_pred[idx] = np.sign(np.dot(self.theta, X_test[idx,:]) + self.theta0)
        #end for
        return y_pred

if __name__ == "__main__":
    
    #load dataset
    iris = load_iris()
    X = iris.data[:100,:]
    y = 2*iris.target[:100] - 1 #map to {+1,-1} labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #perceptron (binary) classifier
    clf = perceptron(num_epochs=5, dim=X.shape[1])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cmt = confusion_matrix(y_test, y_pred)
    acc = np.trace(cmt)/np.sum(np.sum(cmt))
    print("percepton accuracy: ", acc)
    
    #generate plots
    plt.figure()
    sns.heatmap(cmt, annot=True, fmt="d")
    plt.title("Confusion Matrix"); plt.xlabel("predicted"); plt.ylabel("actual")
    #plt.savefig("./figures/perceptron_acc.png")
    plt.show()

    

