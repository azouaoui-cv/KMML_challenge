import csv
import numpy as np
import os
from utils.data import load_data, save_results, compute_kmers_list, K1, κ, P
from utils.models import SVM, SPR
from utils.kernels import GaussianKernel
from utils import FILES, DATA_DIR, RESULT_DIR
import scipy as sp


# DEFINE PARAMETERS
# embeddings parameters
k = 10
σ = 0.4
n_anchors = 6000
# SVM + Gaussian kernel parameters
γ = 500
λ = 6e-6
# choose random seed
np.random.seed(1702)

for q in range(3):
    ####################### COMPUTE EMBEDDINGS ########################
    # choose random anchors
    kmers = compute_kmers_list(q, k)
    index = np.random.choice(range(len(kmers)), replace=False, size = n_anchors)
    anchors = kmers[index]

    # compute K_ZZ
    Z = anchors
    p = len(anchors)
    K_zz = np.zeros((p,p))
    for j in range(p):
        for i in range(j+1):
            K_zz[i,j] = K1(Z[i],Z[j], σ)
    K_zz =  K_zz + K_zz.T
    np.fill_diagonal(K_zz, np.diagonal(K_zz)/2)
    # Then, compute K_ZZ inv**0.5
    β = 1e-3
    print("start matrix inversion")
    K_ZZ_inv_sqr = sp.linalg.sqrtm(sp.linalg.inv(K_zz + β*np.eye(np.shape(K_zz)[0])))

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

    # compute embeddings
    print("start compute embeddings")
    X_train, Y_train, X_test = load_data(q, data_dir=DATA_DIR, files_dict=FILES, mat = False)
    embed_train = []
    for x in X_train:
        embed_train.append(ψ_optim(x,anchors,k,σ))
    E_train = np.array(embed_train)

    #################################################################
    # Do SVM on embeddings

    # Do SVM + Gaussian Kernel predictions
    kernel = GaussianKernel(γ)
    results = np.zeros(3000)

    X_train, Y_train, X_test = load_data(q, data_dir=DATA_DIR, files_dict=FILES)
    clf = SVM(_lambda=λ, kernel=kernel)
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    results[q*1000:q*1000 + 1000] = y_pred


# SAVE Results
save_results("results_test.csv", results, RESULT_DIR)
