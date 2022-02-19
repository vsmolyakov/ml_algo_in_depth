import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class TreeNode():
    def __init__(self, gini, num_samples, num_samples_class, class_label):
        self.gini = gini  #gini cost
        self.num_samples = num_samples #size of node
        self.num_samples_class = num_samples_class #number of node pts with label k
        self.class_label = class_label #predicted class label
        self.feature_idx = 0 #idx of feature to split on
        self.treshold = 0  #best threshold to split on
        self.left = None #left subtree pointer
        self.right = None #right subtree pointer

class DecisionTreeClassifier():
    def __init__(self, max_depth = None):
        self.max_depth = max_depth
    
    def best_split(self, X_train, y_train):
        m = y_train.size
        if (m <= 1):
            return None, None
        
        #number of points of class k
        mk = [np.sum(y_train == k) for k in range(self.num_classes)]

        #gini of current node
        best_gini = 1.0 - sum((n / m) ** 2 for n in mk)
        best_idx, best_thr = None, None

        #iterate over all features
        for idx in range(self.num_features):
            # sort data along selected feature
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

            num_left = [0]*self.num_classes
            num_right = mk.copy()

            #iterate overall possible split positions
            for i in range(1, m):
                
                k = classes[i-1]
                
                num_left[k] += 1
                num_right[k] -= 1

                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.num_classes)
                )

                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.num_classes)
                )

                gini = (i * gini_left + (m - i) * gini_right) / m

                # check that we don't try to split two pts with identical values
                if thresholds[i] == thresholds[i - 1]:
                    continue

                if (gini < best_gini):
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint
                #end if
            #end for
        #end for
        return best_idx, best_thr
    
    def gini(self, y_train):
        m = y_train.size
        return 1.0 - sum((np.sum(y_train == k) / m) ** 2 for k in range(self.num_classes))

    def fit(self, X_train, y_train):
        self.num_classes = len(set(y_train))
        self.num_features = X_train.shape[1]
        self.tree = self.grow_tree(X_train, y_train)

    def grow_tree(self, X_train, y_train, depth=0):
        
        num_samples_class = [np.sum(y_train == k) for k in range(self.num_classes)]
        class_label = np.argmax(num_samples_class)
        
        node = TreeNode(
            gini=self.gini(y_train),
            num_samples=y_train.size,
            num_samples_class=num_samples_class,
            class_label=class_label,
        )

        # split recursively until maximum depth is reached
        if depth < self.max_depth:
            idx, thr = self.best_split(X_train, y_train)
            if idx is not None:
                indices_left = X_train[:, idx] < thr
                X_left, y_left = X_train[indices_left], y_train[indices_left]
                X_right, y_right = X_train[~indices_left], y_train[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self.grow_tree(X_left, y_left, depth + 1)
                node.right = self.grow_tree(X_right, y_right, depth + 1)

        return node

    def predict(self, X_test):
        return [self.predict_helper(x_test) for x_test in X_test]

    def predict_helper(self, x_test):
        node = self.tree
        while node.left:
            if x_test[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.class_label


if __name__ == "__main__":

    #load data
    iris = load_iris()
    X = iris.data[:, [2,3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
          
    print("decision tree classifier...")
    tree_clf = DecisionTreeClassifier(max_depth = 3)
    tree_clf.fit(X_train, y_train)

    print("prediction...")
    y_pred = tree_clf.predict(X_test)
    
    tree_clf_acc = accuracy_score(y_test, y_pred)
    print("test set accuracy: ", tree_clf_acc)
