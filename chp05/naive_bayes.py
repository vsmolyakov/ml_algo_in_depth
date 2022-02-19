import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

from time import time
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

sns.set_style("whitegrid")
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
stop_words.update(['s','t','m','1','2'])

class naive_bayes:
    def __init__(self, K, D):
        self.K = K #number of classes
        self.D = D #dictionary size

        self.pi = np.ones(K) #class priors
        self.theta = np.ones((self.D, self.K)) #bernoulli parameters

    def fit(self, X_train, y_train):
        
        num_docs = X_train.shape[0]
        for doc in range(num_docs):
            
            label = y_train[doc]
            self.pi[label] += 1

            for word in range(self.D):
                if (X_train[doc][word] > 0):
                    self.theta[word][label] += 1
                #end if
            #end for
        #end for
        
        #normalize pi and theta
        self.pi = self.pi/np.sum(self.pi)
        self.theta = self.theta/np.sum(self.theta, axis=0)

    def predict(self, X_test):

        num_docs = X_test.shape[0]
        logp = np.zeros((num_docs,self.K))
        for doc in range(num_docs):
            for kk in range(self.K):
                logp[doc][kk] = np.log(self.pi[kk])
                for word in range(self.D):
                    if (X_test[doc][word] > 0):
                        logp[doc][kk] += np.log(self.theta[word][kk])
                    else:
                        logp[doc][kk] += np.log(1-self.theta[word][kk])
                    #end if
                #end for
            #end for
        #end for
        return np.argmax(logp, axis=1)
        
if __name__ == "__main__":

    #load data
    print("loading 20 newsgroups dataset...")
    tic = time()
    classes = ['sci.space', 'comp.graphics', 'rec.autos', 'rec.sport.hockey']
    dataset = fetch_20newsgroups(shuffle=True, random_state=0, remove=('headers','footers','quotes'), categories=classes)
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.5, random_state=0)
    toc = time()
    print("elapsed time: %.4f sec" %(toc - tic))
    print("number of training docs: ", len(X_train))
    print("number of test docs: ", len(X_test))

    print("vectorizing input data...")
    cnt_vec = CountVectorizer(tokenizer=tokenizer.tokenize, analyzer='word', ngram_range=(1,1), max_df=0.8, min_df=2, max_features=1000, stop_words=stop_words)
    cnt_vec.fit(X_train)
    toc = time()
    print("elapsed time: %.2f sec" %(toc - tic))
    vocab = cnt_vec.vocabulary_
    idx2word = {val: key for (key, val) in vocab.items()}
    print("vocab size: ", len(vocab))

    X_train_vec = cnt_vec.transform(X_train).toarray()
    X_test_vec = cnt_vec.transform(X_test).toarray()

    print("naive bayes model MLE inference...")
    K = len(set(y_train)) #number of classes
    D = len(vocab) #dictionary size
    nb_clf = naive_bayes(K, D)
    nb_clf.fit(X_train_vec, y_train)

    print("naive bayes prediction...")
    y_pred = nb_clf.predict(X_test_vec)
    nb_clf_acc = accuracy_score(y_test, y_pred)
    print("test set accuracy: ", nb_clf_acc)