import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.special import digamma, gammaln

np.random.seed(42)

class LDA:
    def __init__(self, A, K):
        self.N = A.shape[0]  # word (dictionary size)
        self.D = A.shape[1]  # number of documents
        self.K = num_topics  # number of topics

        self.A = A  #term-document matrix

        #init word distribution beta
        self.eta = np.ones(self.N) #uniform dirichlet prior on words
        self.beta = np.zeros((self.N, self.K)) #NxK topic matrix
        for k in range(self.K):
            self.beta[:,k] = np.random.dirichlet(self.eta)
            self.beta[:,k] = self.beta[:,k] + 1e-6 #to avoid zero entries
            self.beta[:,k] = self.beta[:,k]/np.sum(self.beta[:,k])
        #end for

        #init topic proportions theta and cluster assignments z
        self.alpha = np.ones(self.K) #uniform dirichlet prior on topics
        self.z = np.zeros((self.N, self.D)) #cluster assignments z_{id}
        for d in range(self.D):
            theta = np.random.dirichlet(self.alpha)
            wdn_idx = np.nonzero(self.A[:,d])[0]
            for i in range(len(wdn_idx)):
                z_idx = np.argmax(np.random.multinomial(1, theta))
                self.z[wdn_idx[i],d] = z_idx  #topic id
            #end for
        #end for

        #init variational parameters
        self.gamma = np.ones((self.D, self.K)) #topic proportions
        for d in range(self.D):
            theta = np.random.dirichlet(self.alpha)
            self.gamma[d,:] = theta
        #end for

        self.lmbda = np.transpose(self.beta) #np.ones((self.K, self.N))/self.N #word frequencies

        self.phi = np.zeros((self.D, self.N, self.K)) #assignments
        for d in range(self.D):
            for w in range(self.N):
                theta = np.random.dirichlet(self.alpha)
                self.phi[d,w,:] = np.random.multinomial(1, theta)
            #end for
        #end for

    def variational_inference(self):    
             
        var_iter = 10
        llh = np.zeros(var_iter)
        llh_delta = np.zeros(var_iter)

        for iter in range(var_iter):
            print("VI iter: ", iter)
            J_old = self.elbo_objective()
            self.mean_field_update()
            J_new = self.elbo_objective()
            
            llh[iter] = J_old
            llh_delta[iter] = J_new - J_old
        #end for

        #update alpha and beta
        for k in range(self.K):
            self.alpha[k] = np.sum(self.gamma[:,k])            
            self.beta[:,k] = self.lmbda[k,:] / np.sum(self.lmbda[k,:])
        #end for

        #update topic assignments
        for d in range(self.D):
            wdn_idx = np.nonzero(self.A[:,d])[0]
            for i in range(len(wdn_idx)):
                z_idx = np.argmax(self.phi[d,wdn_idx[i],:])
                self.z[wdn_idx[i],d] = z_idx  #topic id
            #end for
        #end for

        plt.figure()
        plt.plot(llh); plt.title('LDA VI');
        plt.xlabel('mean field iterations'); plt.ylabel("ELBO")
        plt.show()

        return llh

    def mean_field_update(self):

        ndw = np.zeros((self.D, self.N)) #word counts for each document
        for d in range(self.D):
            doc = self.A[:,d]
            wdn_idx = np.nonzero(doc)[0]

            for i in range(len(wdn_idx)):
                ndw[d,wdn_idx[i]] += 1
            #end for

            #update gamma
            for k in range(self.K):
                self.gamma[d,k] = self.alpha[k] + np.dot(ndw[d,:], self.phi[d,:,k])
            #end for
            
            #update phi
            for w in range(len(wdn_idx)):
                self.phi[d,wdn_idx[w],:] = np.exp(digamma(self.gamma[d,:]) - digamma(np.sum(self.gamma[d,:])) + digamma(self.lmbda[:,wdn_idx[w]]) - digamma(np.sum(self.lmbda, axis=1)))
                if (np.sum(self.phi[d,wdn_idx[w],:]) > 0): #to avoid 0/0
                    self.phi[d,wdn_idx[w],:] = self.phi[d,wdn_idx[w],:] / np.sum(self.phi[d,wdn_idx[w],:]) #normalize phi
                #end if
            #end for

        #end for
        
        #update lambda given ndw for all docs
        for k in range(self.K):
            self.lmbda[k,:] = self.eta 
            for d in range(self.D):
                self.lmbda[k,:] += np.multiply(ndw[d,:], self.phi[d,:,k])
            #end for
        #end for

    def elbo_objective(self):
        #see Blei 2003

        T1_A = gammaln(np.sum(self.alpha)) - np.sum(gammaln(self.alpha))
        T1_B = 0
        for k in range(self.K):
            T1_B +=  np.dot(self.alpha[k]-1, digamma(self.gamma[:,k]) - digamma(np.sum(self.gamma, axis=1)))
        T1 = T1_A + T1_B
        
        T2 = 0
        for n in range(self.N):
            for k in range(self.K):
                T2 += self.phi[:,n,k] * (digamma(self.gamma[:,k]) - digamma(np.sum(self.gamma, axis=1)))

        T3 = 0
        for n in range(self.N):
            for k in range(self.K):
                T3 += self.phi[:,n,k] * np.log(self.beta[n,k])

        T4 = 0
        T4_A = -gammaln(np.sum(self.gamma, axis=1)) + np.sum(gammaln(self.gamma), axis=1)
        T4_B = 0
        for k in range(self.K):
            T4_B = -(self.gamma[:,k]-1) * (digamma(self.gamma[:,k]) - digamma(np.sum(self.gamma, axis=1)))
        T4 = T4_A + T4_B
        
        T5 = 0
        for n in range(self.N):
            for k in range(self.K):
                T5 += -np.multiply(self.phi[:,n,k], np.log(self.phi[:,n,k] + 1e-6))

        T15 = T1 + T2 + T3 + T4 + T5
        J = sum(T15)/self.D  #averaged over documents
        return J 

if __name__ == "__main__":

    #LDA parameters
    num_features = 1000  #vocabulary size
    num_topics = 4      #fixed for LD

    #20 newsgroups dataset
    categories = ['sci.crypt', 'comp.graphics', 'sci.space', 'talk.religion.misc']
    
    newsgroups = fetch_20newsgroups(shuffle=True, random_state=42, subset='train',
                 remove=('headers', 'footers', 'quotes'), categories=categories)
    
    vectorizer = TfidfVectorizer(max_features = num_features, max_df=0.95, min_df=2, stop_words = 'english')
    dataset = vectorizer.fit_transform(newsgroups.data)
    A = np.transpose(dataset.toarray())  #term-document matrix

    lda = LDA(A=A, K=num_topics)
    llh = lda.variational_inference()
    id2word = {v:k for k,v in vectorizer.vocabulary_.items()}

    #display topics
    for k in range(num_topics):
        print("topic: ", k)
        print("----------")
        top_words = np.argsort(lda.lmbda[k,:])[-5:]
        for i in range(len(top_words)):
            print(id2word[top_words[i]])
