# ML Algorithms From Scratch
ML Algorithms From Scratch: Bayesian Inference and Deep Learning

**Chp02: Markov Chain Monte Carlo (MCMC)**
- [Estimate Pi](./chp02/monte_carlo_pi.py): Monte Carlo estimate of Pi
- [Binomial Tree Model](./chp02/binomial_tree.py): Monte Carlo simulation of binomial stock price
- [Random Walk](./chp02/random_walk.py): self-avoiding random walk
- [Gibbs Sampling](./chp02/gibbs_gauss.py): Gibbs sampling of multivariate Gaussian distribution
- [Metropolis-Hastings Sampling](./chp02/mh_gauss2d.py): Metropolis-Hastings sampling of multivariate Gaussian mixture
- [Importance Sampling](./chp02/imp_samp.py): importance sampling for finding expected value of a function

**Chp03: Variational Inference (VI)**
- [Mean Field VI](./chp03/mean_field_mrf.py): image denoising in Ising model

**Chp04: Software Implementation**
- [Subset Generation](./chp04/subset_gen.py): a complete search algorithm
- [Fractional Knapsack](./chp04/knapsack_greedy.py): a greedy algorithm
- [Binary Search](./chp04/binary_search.py): a divide and conquer algorithm
- [Binomial Coefficients](./chp04/binomial_coeffs.py): a dynamic programming algorithm

**Chp05: Classification Algorithms**
- [Perceptron](./chp05/perceptron.py): perceptron algorithm
- [SVM](./chp05/svm.py): support vector machine
- [SGD-LR](./chp05/sgd_lr.py): stochastic gradient descent logistic regression
- [Naive Bayes](./chp05/naive_bayes.py): Bernoulli Naive Bayes algorithm
- [CART](./chp05/cart.py): decision tree classification algorithm

**Chp06: Regression Algorithms**
- [KNN](./chp06/knn_reg.py): K-Nearest Neighbors regression
- [BLR](./chp06/ridge_reg.py): Bayesian linear regression
- [HBR](./chp06/hierarchical_regression.py): Hierarchical Bayesian regression
- [GPR](./chp06/gp_reg.py): Gaussian Process regression

**Chp07: Selected Supervised Learning Algorithms**
- [Page Rank](./chp07/page_rank.py): Google page rank algorithm
- [HMM](./chp07/hmm.py): EM algorithm for Hidden Markov Models
- Imbalanced Learning: [Tomek Links](./chp07/plot_tomek_links.py), [SMOTE](./chp07/plot_smote_regular.py)
- Active Learning: [LR](./chp07/demo_logreg.py)
- Bayesian optimization: [BO](./chp07/bayes_opt_sklearn.py)
- Ensemble Learning: [Bagging](./chp07/bagging_clf.py), [Boosting](./chp07/adaboost_clf.py), [Stacking](./chp07/stacked_clf.py)

**Chp08: Unsupervised Learning Algorithms**
- [DP-Means](./chp08/dpmeans.py): Dirichlet Process (DP) K-Means
- [EM-GMM](./chp08/gmm.py): EM algorithm for Gaussian Mixture Models
- [PCA](./chp08/pca.py): Principal Component Analysis
- [t-SNE](./chp08/manifold_learning.py): t-SNE manifold learning

**Chp09: Selected Unsupervised Learning Algorithms**
- [LDA](./chp09/lda.py): Variational Inference for Latent Dirichlet Allocation
- [KDE](./chp09/kde.py): Kernel Density Estimator
- [TPO](./chp09/portfolio_opt.py): Tangent Portfolio Optimization
- [ICE](./chp09/inv_cov.py): Inverse Covariance Estimation
- [SA](./chp09/sim_annealing.py): Simulated Annealing
- [GA](./chp09/ga.py): Genetic Algorithm

**Chp10: Fundamental Deep Learning Algorithms**
- [MLP](./chp10/mlp.py): Multi-Layer Perceptron
- [LeNet](./chp10/lenet.py): LeNet for MNIST digit classification
- [ResNet](./chp10/image_search.py): ResNet50 image search on CalTech101 dataset
- [LSTM](./chp10/lstm_sentiment.py): LSTM sentiment classification of IMDB movie dataset
- [MINN](./chp10/multi_input_nn.py): Mult-Input Neural Net model for sequence similarity of Quora question pairs dataset
- [OPT](./chp10/keras_optimizers.py): Neural Net Optimizers

**Chp11: Advanced Deep Learning Algorithms**
- [LSTM-VAE](./chp11/lstm_vae.py): time-series anomaly detector
- [MDN](./chp11/keras_mdn.py): mixture density network
- [Transformer](./chp11/transformer.py): for text classification
- [GNN](./chp11/spektral_gnn.py): graph neural network

**Environment**

To install required libraries, please run the following commands:

```
python3 -m venv ml-algo

source ml-algo/bin/activate    //in linux
.\ml-algo\Scripts\activate.bat //in CMD windows
.\ml-algo\Scripts\Activate.ps1 //in Powershell windows

pip install -r requirements.txt
```


