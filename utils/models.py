"""

Implement some binary classifiers including:

- Simple Patter Recognition (SPR)
- Support Vector Machines (SVM)

"""

###########
# Imports #
###########
import numpy as np
from scipy import optimize
from tqdm import tqdm


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
        
        self.X_train = X
        self.X0 = X[y == 0]
        self.X1 = X[y == 1]
        
        self.m0 = len(self.X0)
        self.m1 = len(self.X1)
        
        if self.kernel == False:
            self.b = 1/2 * (1/(self.m0**2)*np.sum(self.X0.dot(self.X0.T)) 
                            - 1/(self.m1**2)*np.sum(self.X1.dot(self.X1.T)))
        else:
            # à changer (comment ?)
            self.list0 = list(np.where(y==0)[0])
            self.list1 = list(np.where(y==1)[0])
            self.b = 1/2 * (1/(self.m0**2)*np.sum([self.kernel(self.X_train[i],self.X_train[j]) 
                                                   for i in self.list0 for j in self.list0])
                            - (1/(self.m1**2))*np.sum([self.kernel(self.X_train[i],self.X_train[j]) 
                                                       for i in self.list1 for j in self.list1]))

    
    def predict(self,X):
        
        y_pred = np.zeros(len(X))
        
        for i in range(len(X)):
            if self.kernel == False:
                val = (1 / self.m1) * np.sum(self.X1.dot(X[i])) - (1 / self.m0) * np.sum(self.X0.dot(X[i])) + self.b
            else:
                val = ((1/self.m1)*np.sum([self.kernel(self.X_train[k],X[i]) for k in self.list1]) 
                       - (1/self.m0)*np.sum([self.kernel(self.X_train[k],X[i]) for k in self.list0])) + self.b                    
            y_pred[i] = np.sign(val)/2 + 1/2
        return y_pred    
    
    
    def score(self, y, y_pred):
        return np.sum([y == y_pred]) / len(y)
    
    


class PCA():
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
