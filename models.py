"""

Implement some binary classifiers including:

- Simple Patter Recognition (SPR)
- Support Vector Machines (SVM)

And other methods:

- Kernel PCA
- K-means
- Spectral clustering

"""

###########
# Imports #
###########
import numpy as np
from scipy import optimize


class SVM:
    """
    This class implements the Support Vector Machine algorithm
    """
    def __init__(self, _lambda, kernel, maxiter=15000):
        self.kernel = kernel
        self._lambda = _lambda
        self.maxiter = maxiter


    def fit(self,X,y):
        """
        Fitting phase

        Parameters
        ------------
        - X : numpy.ndarray
            Data matrix

        - y : numpy.array
            Labels
        """

        self.X_train = X

        # Define the kernel matrix K
        K = self.kernel.compute_gram_matrix(self.X_train)

        # transpose Y_train to fit the optimization formulation
        y = y * 2 - 1

        # Use scipy.optimize to solve the problem
        self.n = len(y)

        # Define the loss function
        f = lambda x: 1/2 * x.T.dot(K).dot(x) - y.T.dot(x)

        # Define the jacobian of the loss function
        grad_f = lambda x: K.dot(x) - y

        # Define the bounds (sequences of min, max) (This depends on the sign of Y_train)
        bounds = [[0, y[i] / (2 * self.n * self._lambda)]
                  if y[i] > 0
                  else [y[i] / (2 * self.n * self._lambda), 0]
                  for i in range(self.n)]

        # define initial point
        x0 = np.zeros(self.n)

        # define number max iteration
        opts = {"maxiter": self.maxiter}

        # optimize
        res = optimize.minimize(f, x0, jac=grad_f, bounds=bounds, method="L-BFGS-B", options=opts)

        # save results
        self.alpha = res["x"]

    def predict(self, X):
        """
        Prediction phase

        Parameters
        -----------
        - X : numpy.ndarray
            Testing data

        Returns
        -----------
        - y_pred : numpy.array
            Prediction array
        """
        # Compute the similarity matrix for faster prediction
        S = self.kernel.compute_similarity_matrix(X)
        y_pred = S.dot(self.alpha)

        # Rescale the prediction to 0 and 1
        y_pred = np.sign(y_pred) / 2 + 1/2

        return y_pred


    def score(self, y, y_pred):
        return np.sum([y == y_pred]) / len(y)

class SVM_precomputed_gram:
    """
    This class implements the Support Vector Machine algorithm
    using a pre-defined Gram Matrix (useful when tuning SVM+ConvKernel)
    """
    def __init__(self, _lambda, kernel, maxiter=15000):
        self.kernel = kernel
        self._lambda = _lambda
        self.maxiter = maxiter


    def fit(self,X,y, K):
        """
        Fitting phase

        Parameters
        ------------
        - X : numpy.ndarray
            Data matrix

        - y : numpy.array
            Labels
        """

        self.X_train = X

        # transpose Y_train to fit the optimization formulation
        y = y * 2 - 1

        # Use scipy.optimize to solve the problem
        self.n = len(y)

        # Define the loss function
        f = lambda x: 1/2 * x.T.dot(K).dot(x) - y.T.dot(x)

        # Define the jacobian of the loss function
        grad_f = lambda x: K.dot(x) - y

        # Define the bounds (sequences of min, max) (This depends on the sign of Y_train)
        bounds = [[0, y[i] / (2 * self.n * self._lambda)]
                  if y[i] > 0
                  else [y[i] / (2 * self.n * self._lambda), 0]
                  for i in range(self.n)]

        # define initial point
        x0 = np.zeros(self.n)

        # define number max iteration
        opts = {"maxiter": self.maxiter}

        # optimize
        res = optimize.minimize(f, x0, jac=grad_f, bounds=bounds, method="L-BFGS-B", options=opts)

        # save results
        self.alpha = res["x"]

    def predict(self, X, S):
        """
        Prediction phase

        Parameters
        -----------
        - X : numpy.ndarray
            Testing data

        Returns
        -----------
        - y_pred : numpy.array
            Prediction array
        """
        # Compute the similarity matrix for faster prediction
        y_pred = S.dot(self.alpha)

        # Rescale the prediction to 0 and 1
        y_pred = np.sign(y_pred) / 2 + 1/2

        return y_pred


    def score(self, y, y_pred):
        return np.sum([y == y_pred]) / len(y)


class SPR:
    """
    This class implements the Simple Pattern Recognition algorithm found in the Learning with Kernel books
    """
    def __init__(self, kernel=False):
        self.kernel = kernel

    def fit(self,X,y):
        """
        Fitting phase

        Parameters
        ------------
        - X : numpy.ndarray
            Data matrix

        - y : numpy.array
            Labels
        """

        self.X0 = X[y == 0]
        self.X1 = X[y == 1]
        self.m0 = len(self.X0)
        self.m1 = len(self.X1)

        K0 = self.kernel.compute_gram_matrix(self.X0)
        K1 = self.kernel.compute_gram_matrix(self.X1)
        self.b = 1/2 * (1/(self.m0**2)*np.sum(K0)- 1/(self.m1**2)*np.sum(K1))

    def predict(self,X):

        S1 = self.kernel.compute_similarity_matrix(X, self.X1)
        S0 = self.kernel.compute_similarity_matrix(X, self.X0)

        val = (1/self.m1*np.sum(S1, axis = 1) - 1/self.m0*np.sum(S0, axis = 1)) + self.b

        y_pred = np.sign(val)/2 + 1/2

        return y_pred


    def score(self, y, y_pred):
        return np.sum([y == y_pred]) / len(y)




class PCA:
    """
    This class implement the Kernel PCA
    """

    def __init__(self, kernel):
        self.kernel = kernel



    def fit(self,X):
        """
        Fitting phase

        Parameters
        ------------
        - X : numpy.ndarray
            Data matrix
        """

        self.X_train = X
        K = self.kernel.compute_gram_matrix(self.X_train)
        n = np.shape(K)[0]

        # 1) Center the Gram matrix
        U = (1/n)*np.ones((n,n))
        I = np.eye(n)
        Kc =  (I-U).dot(K).dot(I-U)

        # 2) Compute the first eigenvectors (ui, ∆i)
        _lambda, v = np.linalg.eig(Kc)

        # 3) Normalize the eigenvectors αi = ui/√∆i
        alpha = v/_lambda

        self._lambda = np.real(_lambda)
        self.alpha = np.real(alpha)


    def proj(self, X, n):

        """
        This function implement the projection on principal axis of  the Kernel PCA

        Parameters
        ------------
        - X : numpy.ndarray
        Points to be projected
        - n : int
        Number of principal axis
        - kernel : function
        kernel of the PCA
        """

        K = self.kernel.compute_similarity_matrix(X)
        X_projected = K.dot(self.alpha[:,:n])

        return X_projected



def Kmeans(X, K, max_iter, tol=1e-30):

    #step 0 (initialize centroids mu)
    idx = np.random.randint(len(X), size=K)
    mu = X[idx]

    n_iter = 0
    stop = False
    d = 1e6
    while stop != True:

        #create clusters
        clustering = np.zeros(len(X))

        #step 1 (minimizing by assigning a cluster to each point)
        for i in range(len(X)):
            clustering[i] = np.argmin(np.linalg.norm(X[i]-mu, axis=1))


        #step 2 (minimizing w.r.t mu)
        for k in range(K):
            if np.sum([clustering==k]) != 0:
                mu[k] = np.mean(X[clustering==k], axis=0)


        d_new = distortion(X, mu, clustering)
        #print(d_new)

        if np.abs(d_new-d) < tol or n_iter > max_iter:
            stop = True

        d = d_new

        n_iter +=1



    return mu, clustering


def distortion(X, mu, clustering):
    dis = 0
    for k in range(len(mu)):
        dis = dis + np.linalg.norm(X[clustering==k] - mu[k])**2
    return dis


# We try several random initializations and keep the partition which minimize the distorsion.
def Kmeans_try(X, n_try, n_cluster, max_iter):

    for i in range(n_try):

        mu, cl = Kmeans(X, n_cluster, max_iter)

        if i == 0:
            dist_min = distortion(X, mu, cl)
            mu_min, cl_min = mu, cl
        else:
            if distortion(X, mu, cl) < dist_min:
                dist_min = distortion(X, mu, cl)
                mu_min, cl_min = mu, cl


    return mu_min, cl_min, dist_min


def spec_cl(n_cl, kmers, σ):

    '''Spectral Clustering'''

    # compute Gram matrix
    k = int(len(kmers[0])/4)
    kernel = GaussianKernel(1/(2*(σ**2)*k))
    K = k*kernel.compute_gram_matrix(kmers)

    # Compute the n_cl first eigenvectors (ui, ∆i)
    λ, v = sp.linalg.eigh(K)

    # compute the maximum entry of a row
    #cluster_idx = np.argmax(v[:,:n_cl], axis = 1)
    # OR Kmeans on the rows
    n_try = 1
    max_iter = 20
    Z = v[:,:n_cl]/np.linalg.norm(v[:,:n_cl],axis=1).reshape(-1,1) # normalize v
    mu_rows, cluster_idx, dist = Kmeans_try(Z, n_try, n_cl, max_iter)

    # compute the barycenter
    mu = []
    for i in range(n_cl):
        if np.sum([cluster_idx==i]) != 0:
            bary = np.mean(kmers[cluster_idx==i], axis = 0)
            mu.append(bary)

    return mu, v
