"""

utility functions, including:

- P : compute one k-mer
- compute_kmers_list
- κ
- K1
- K_zz_inv_sqr
- compute_embeddings

"""

###########
# Imports #
###########

import csv
import numpy as np
import scipy as sp
import os
from utils import FILES, DATA_DIR, RESULT_DIR, LOGGING_DIR, EMBEDDING_DIR
from utils import load_data


def P(i, seq, k, zero_padding=True):
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

def compute_kmers_list(idx, k):

    """
    This function compute all the k-mers of a list of sequences

    Parameters
    ------------
    - idx : int
        index of the dataset (0,1, or 2)
    - k : int
        length of the k-mers
    """

    X_train, Y_train, X_test = load_data(idx, data_dir=DATA_DIR, files_dict=FILES, mat = False)
    n = len(X_train)
    m = len(X_train[0])

    kmers = []
    for x in X_train:
        for i in range(m):
            p = P(i,x,k)
            if p[1] == False:
                kmers.append(p[0])

    kmers = np.array(kmers)

    return kmers

def κ(u, σ):
    return np.exp((u-1)/σ**2)

def K1(z1, z2, σ):
    z1_norm = np.linalg.norm(z1)
    z2_norm = np.linalg.norm(z2)
    z1z2_norm = z1_norm*z2_norm
    u = z1.dot(z2)/z1z2_norm
    return z1z2_norm*κ(u,σ)

def K_zz_inv_sqr(anchors, σ, β=1e-3):
    """
    This function compute the root square inverse of the matrix of anchors

    Parameters
    ------------
    - anchors : numpy.array
        matrix with the anchors as rows
    - σ : float
        parameters in the exponential function
    - k : int
        length of the k-mers
    """
    # compute K_ZZ
    Z = anchors
    p = len(anchors)
    K_zz = np.zeros((p,p))
    for j in range(p):
        for i in range(j+1):
            K_zz[i,j] = K1(Z[i],Z[j], σ)
    Kκ_zz =  K_zz + K_zz.T
    np.fill_diagonal(K_zz, np.diagonal(K_zz)/2)

    # Then, compute K_ZZ inv**0.5
    print("start matrix inversion", flush=True)
    K_ZZ_inv_sqr = sp.linalg.inv(sp.linalg.sqrtm(K_zz + β*np.eye(np.shape(K_zz)[0])))

    return K_ZZ_inv_sqr


def compute_embeddings(X, Z_anchor, k, σ, K_ZZ_inv_sqr):

    """
    This function compute the embeddings of a matrix of points X

    Parameters
    ------------
    - X : numpy.array
        matrix with the data points as rows
    - Z_anchor : numpy.array
        list of the anchors as the rows of Z_anchor
    - k : int
        length of the k-mers
    - σ : int
        parameters in the exponential function
    - K_ZZ_inv_sqr : numpy.array
        square root inverse of K_zz
    """

    # define approximate mapping thanks to the anchors
    def ψ_optim(x, Z_anchor, k , σ):
        P_x = np.array([P(i,x,k)[0] for i in range(len(x)) if P(i,x,k)[1] == False])
        Z = np.array(Z_anchor)
        Z = Z/np.linalg.norm(Z,axis=1).reshape(-1,1) # normalize Z rows
        P_x_norm = P_x/np.linalg.norm(P_x,axis=1).reshape(-1,1) # normalize P_x rows
        S = Z.dot(P_x_norm.T)
        S = np.einsum('i, ij -> ij',np.linalg.norm(Z,axis=1), np.sqrt(k)*np.exp((S - 1)/σ**2))
        b = K_ZZ_inv_sqr.dot(S)
        return np.sum(b, axis=1)/np.shape(b)[1]

    p = len(Z_anchor)
    assert np.all(K_ZZ_inv_sqr.imag == np.zeros((p, p))), "imaginary coefficients"

    ####################### COMPUTE EMBEDDINGS ########################
    print("start compute embeddings")
    embed = []
    for x in X:
        embed.append(ψ_optim(x, Z_anchor, k, σ))
    embed = np.array(embed)

    return embed
