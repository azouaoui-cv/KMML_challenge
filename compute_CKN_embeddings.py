"""
Compute and store Convolution Kernel Network embeddings
using different hyperparameters
"""

###########
# Imports #
###########
import numpy as np
import os
from utils.data import load_data, compute_kmers_list, K1, P
from utils import FILES, DATA_DIR
import scipy as sp

#########
# Paths #
#########
EMBEDDING_DIR = os.path.join(os.getcwd(), "embeddings")

if not os.path.isdir(EMBEDDING_DIR):
    os.mkdir(EMBEDDING_DIR)

#########
# Setup #
#########
# DEFINE embeddings parameters lists
k_list = [9, 10, 11]
sigma_list = [0.35, 0.4, 0.45]
n_anchors = 6000
np.random.seed(1702)

########
# Main #
########
if __name__ == "__main__":
    for k in k_list:
        for σ in sigma_list:
            for q in range(3):
                print("params: {0, 1}. dataset: {2}"
                      "".format(k, σ, q))
                # choose random anchors
                kmers = compute_kmers_list(q, k)
                index = np.random.choice(range(len(kmers)), replace=False, size=n_anchors)
                anchors = kmers[index]

                # compute K_ZZ
                Z = anchors
                p = len(anchors)
                K_zz = np.zeros((p, p))
                for j in range(p):
                    for i in range(j+1):
                        K_zz[i,j] = K1(Z[i], Z[j], σ)
                Kκ_zz =  K_zz + K_zz.T
                np.fill_diagonal(K_zz, np.diagonal(K_zz) / 2)
                # Then, compute K_ZZ inv**0.5
                β = 1e-3
                print("start matrix inversion", flush=True)
                K_ZZ_inv_sqr = sp.linalg.inv(sp.linalg.sqrtm(K_zz + β * np.eye(np.shape(K_zz)[0])))

                assert np.all(K_ZZ_inv_sqr.imag == np.zeros((p, p))), "imaginary coefficients"

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
                X_train, Y_train, X_test = load_data(q, data_dir=DATA_DIR, files_dict=FILES, mat=False)
                embed_train = []
                for x in X_train:
                    embed_train.append(ψ_optim(x,anchors,k,σ))
                E_train = np.array(embed_train)

                print(np.shape(E_train))
                # SAVE embeddings
                np.save(os.path.join(EMBEDDING_DIR, 
                                     "embedding_d{0}_s{1}_k{2}.npy"
                                     "".format(q, round(σ, 3), k)),
                        E_train)