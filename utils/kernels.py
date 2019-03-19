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


class ConvKernel(Kernel):
    def __init__(self, sigma, k):
        super().__init__()
        self.sigma = sigma
        self.k = k

    def __call__(self, x, y):
        """
        Convolutional kernel similarity score on two data points
        """
        mx = len(x)
        my = len(y)
        Px = np.array([P(i,x,k)[0] for i in range(mx)])
        Py = np.array([P(i,y,k)[0] for i in range(mx)])
        PxPyt = Px.dot(Py.T)/k
        s = k*np.exp((1/(sigma**2))*(PxPyt-1))
        return np.sum(s)/(mx*my)


    def P(self, i, seq, k, zero_padding=True):
        """
        Compute the a k_mers at a given position in a nucleotides sequence

        Parameters
        -----------
        - i : int
            Position in the sequence

        - k : int
            Size of k-mer to be returned

        - seq : str
            Sequence of nucleotides

        - zero_padding : boolean (optional)
            Whether to use zero-padding on the sequence edges
            Default: True

        Returns
        -----------
        - L : numpy.array
            One-hot encoding of the string sequence

        - not_in : boolean
            Whether the k-mer was computed on the sequence edges
            Always set to False when using zero-padding
        """

        ENCODING = {'A': [1.,0.,0.,0.],
                'C': [0.,1.,0.,0.],
                'G': [0.,0.,1.,0.],
                'T': [0.,0.,0.,1.],
                'Z': [0.,0.,0.,0.]} # used in zero-padding


        # Setup
        not_in = True
        if zero_padding:
            not_in = False

        # lower edge
        if i-(k+1)//2 + 1 < 0:
            # Use heading zero padding here
            n_zeros = abs(i - (k+1) // 2 + 1)
            k_mer_i = 'Z'*n_zeros + seq[:  i + (k+2)//2]
        # upper edge
        elif i + (k+2)//2 > len(seq):
            # Use trailing zero padding here
            n_zeros = i + (k+2) // 2 - len(seq)
            k_mer_i = seq[i - (k+1)//2 + 1:] + 'Z'*n_zeros
        # in the middle
        else:
            k_mer_i = seq[i-(k+1)//2 + 1 :  i + (k+2)//2]
            not_in = False

        # concatenate one hot encoding
        L = []
        for c in k_mer_i:
            L += ENCODING[c]

        # Sanity check
        assert len(L) == 4 * k

        # Convert to array and return
        return np.array(L), not_in


    def compute_gram_matrix(self, X):
        """
        Compute the Convolutional kernel Gram matrix
        """
        self.X = X

        # compute list of kmers
        P_list = []
        for i in range(len(X)):
            x = X[i]
            mx = len(x)
            Px = np.array([self.P(i,x,self.k)[0] for i in range(mx)])
            P_list.append(Px)

        n = len(X)
        K = np.zeros((n,n))
        print('ok')
        for i in range(n):
            if(i%(n//10)==0):
                print((10*i)//n, end= '% ')
            for j in range(i+1):
                x = X[i]
                y = X[j]
                mx = len(x)
                my = len(y)
                PxPyt = P_list[i].dot(P_list[j].T)/self.k
                s = self.k*np.exp((1/(self.sigma**2))*(PxPyt-1))
                K[i,j] = np.sum(s)/(mx*my)
        for i in range(n):
            for j in range(i+1,n):
                K[i,j] = K[j,i]

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

        # compute lists of kmers
        P_list_X = []
        for i in range(len(self.X)):
            x = self.X[i]
            mx = len(x)
            Px = np.array([self.P(i,x,self.k)[0] for i in range(mx)])
            P_list_X.append(Px)
        P_list_Z = []
        for i in range(len(Z)):
            z = Z[i]
            mz = len(z)
            Pz = np.array([self.P(i,z,self.k)[0] for i in range(mz)])
            P_list_Z.append(Pz)


        n = len(self.X)
        m = len(Z)
        K = np.zeros((m,n))
        for i in range(m):
            if(i%(m//10)==0):
                print((10*i)//m, end= '% ')
            for j in range(n):
                z = Z[i]
                x = self.X[j]
                mz = len(z)
                mx = len(x)
                PzPxt = P_list_Z[i].dot(P_list_X[j].T)/self.k
                s = self.k*np.exp((1/(self.sigma**2))*(PzPxt-1))
                K[i,j] = np.sum(s)/(mx*mz)
        return K
