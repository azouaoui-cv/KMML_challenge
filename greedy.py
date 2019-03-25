import csv
import os
import numpy as np
from scipy import optimize
from itertools import product
import matplotlib.pyplot as plt
import scipy as sp
from time import time
from utils.data import load_data, save_results, P, κ, K1, compute_kmers_list
from utils.models import SVM, SPR, PCA
from utils.kernels import GaussianKernel

idx= 0
σ = 0.36
k = 9
objectif = 10 # number of anchors for one batch
kmers = compute_kmers_list(idx, k)
n_batch = 12
all_index = []
# choose initial random anchor
np.random.seed(1702)

for _ in range(n_batch):

    n0 = np.random.randint(len(kmers))
    anchors = list(kmers[n0:n0+1])

    # init parameters
    n_anchors = len(anchors)
    index_list = [n0]

    # compute K_zz
    p = len(anchors)
    Z = np.array(anchors)
    K_zz = np.zeros((p,p))
    for j in range(p):
        for i in range(j+1):
            K_zz[i,j] = K1(Z[i],Z[j], σ)
    K_zz =  K_zz + K_zz.T
    np.fill_diagonal(K_zz, np.diagonal(K_zz)/2)
    # compute K_zz inv
    K_ZZ_inv = sp.linalg.inv(K_zz)


    for k in range(objectif-1):

        # do data point selection
        Z = np.array(anchors)
        Z = Z/np.linalg.norm(Z,axis=1).reshape(-1,1) # normalize Z rows
        kmers_normalize = kmers/np.linalg.norm(kmers,axis=1).reshape(-1,1) # normalize kmers rows
        S = Z.dot(kmers_normalize.T)
        S = np.exp((S - 1)/σ**2)
        S = np.einsum('i, ij -> ij',np.linalg.norm(Z,axis=1), S)
        S = np.einsum('j, ij -> ij',np.linalg.norm(kmers,axis=1), S)
        b = K_ZZ_inv.dot(S)
        b = np.einsum('ij,ij -> j', S, b)
        norms = np.linalg.norm(kmers, axis=1)
        norms2 = np.multiply(norms, norms)
        new_anchor = kmers[np.argmax(norms2 - b)]

        # update the set of anchors
        anchors.append(new_anchor)
        index_list.append(np.argmax(norms2 - b))

        # compute new K_zz_inv
        f = Z.dot(new_anchor/np.linalg.norm(new_anchor))
        f = np.exp((f - 1)/σ**2).reshape(-1,1)
        f = np.einsum('i, ij -> ij',np.linalg.norm(Z,axis=1), f)
        f = np.einsum('j, ij -> ij',np.linalg.norm(new_anchor.reshape(1,-1),axis=1), f)
        b = K_ZZ_inv.dot(f)
        s = np.linalg.norm(new_anchor)**2 - f.T.dot(K_ZZ_inv.dot(f))
        A = K_ZZ_inv + (1/s)*b.dot(b.T)
        B = -(1/s)*b
        C = -(1/s)*b.T
        D = 1/s
        AB = np.concatenate((A,B), axis=1)
        CD = np.concatenate((C,D), axis=1)
        K_ZZ_inv = np.concatenate((AB,CD), axis=0)

        #if k%(objectif//100) == 0:
        #    print(k//(objectif//100), end='% ')

    all_index += index_list

np.save('index_list.npy', np.array(all_index))

print(np.shape(np.load('index_list.npy')))
