"""

Implement some kernels to be used in the ``models.py`` module, including:

- GaussianKernel
- ...

"""

###########
# Imports #
###########
import numpy as np

################
# Parent class #
################

class Kernel():
    def __init__(self):
        """
        Kernel initialization
        """
        pass
    
    def __call__(self, Xi, Xj):
        """
        Kernel computation
        """
        pass
        
    def compute_gram_matrix(self, X):
        """
        Compute the Gram matrix
        """
        self.X = X
        pass
    
    def compute_similarity_matrix(self, Z, X=None):
        """
        Compute the similarity matrix given data points using the kernel Gram matrix
        
        Parameters
        -------------
        - Z : numpy.ndarray
            Test data matrix
            
        - X : numpy.ndarray (optional)
            Training or subset of training data matrix
            To be used to overwrite the kernel.X matrix
            Default: None (i.e. unused)
        """
        pass

#################
# Child classes #
#################
    
class GaussianKernel(Kernel):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        
    def __call__(self, x, y):
        """
        Gaussian kernel similarity score on two data points
        """
        return np.exp(- self.gamma * np.linalg.norm(x-y)**2)
    
    def compute_gram_matrix(self, X):
        """
        Compute the Gaussian kernel Gram matrix
        """
        self.X = X
        # Compute the squared euclidean norm for each data point
        X2 = np.sum(np.multiply(self.X, self.X), 1)[:, np.newaxis]
        # Compute the squared euclidean norm of pairwise differences using the dot product
        K0 = X2 + X2.T - 2 * self.X.dot(self.X.T)
        # Compute the Gram matrix using numpy vectorized functions
        K = np.power(np.exp(- self.gamma), K0)
        return K
    
    def compute_similarity_matrix(self, Z, X=None):
        """
        Compute the similarity matrix given data points using the kernel Gram matrix
        
        Parameters
        -------------
        - Z : numpy.ndarray
            Test data matrix
            
        - X : numpy.ndarray (optional)
            Training or subset of training data matrix
            To be used to overwrite the kernel.X matrix
            Default: None (i.e. unused)
        """
        if X is not None:
        	self.X = X
        # Compute the squared euclidean norm for each data point (train)
        X2 = np.sum(self.X ** 2, 1)[:, np.newaxis]
        # Compute the squared euclidean norm for each data point (test)
        Z2 = np.sum(Z ** 2, 1)[:, np.newaxis]
        # Compute the squared euclidean norm of pairwise differences using the dot product
        S0 = Z2 + X2.T - 2 * Z.dot(self.X.T)
        # Compute the similarity matrix using numpy vectorized functions
        S = np.power(np.exp(- self.gamma), S0)
        return S
    
    
class LinearKernel(Kernel):
    def __init__(self):
        super().__init__()
        
    def __call__(self, x, y):
        """
        Linear kernel similarity score on two data points
        """
        return x.dot(y)
    
    def compute_gram_matrix(self, X):
        """
        Compute the Gaussian kernel Gram matrix
        """
        self.X = X
        # Compute the Gram matrix using numpy vectorized functions
        K = self.X.dot(self.X.T)
        return K
    
    def compute_similarity_matrix(self, Z, X=None):
        """
        Compute the similarity matrix given data points using the kernel Gram matrix
        
        Parameters
        -------------
        - Z : numpy.ndarray
            Test data matrix
            
        - X : numpy.ndarray (optional)
            Training or subset of training data matrix
            To be used to overwrite the kernel.X matrix
            Default: None (i.e. unused)
        """
        if X is not None:
        	self.X = X
        # Compute the similarity matrix using numpy vectorized functions
        S = Z.dot(self.X.T)
        return S