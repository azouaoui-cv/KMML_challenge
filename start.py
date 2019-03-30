"""
Main script

Do the final prediction for the challenge:

1) Predict with a SVM (gaussian kernel)
2) Predict with a SVM (convolutional kernel)
3) Predict with Convolutional kernel network (cf. article Chen, Jacob, Mairal, "Biological Sequence Modeling
                                                                                    with Convolutional Kernel Networks" )

Do a bagging with the three predictions and save the final prediction as "Yte.csv"

"""


# Import
import csv
import numpy as np
import os
from utils import load_data, save_results, bagging
from utils_CKN import compute_kmers_list, K_zz_inv_sqr, compute_embeddings
from models import SVM
from kernels import GaussianKernel, ConvKernel
from utils import FILES, DATA_DIR, RESULT_DIR


# Paths sanity check
if not os.path.exists(RESULT_DIR):
    os.mkdir(RESULT_DIR)    
if not os.path.exists(DATA_DIR):
    raise ValueError("Please create a ``data`` directory "
                     "containing the data .csv files.")

print("Starting the main script...")
print("It consists in bagging 3 different classifiers outputs...")
print("This can take over an hour...")
################################
# 1) SVM with Gaussian kernel  #
################################
print("1/3 Starting SVM with Gaussian kernel...")
# Define parameters lists
gamma_list = [391, 292, 325]
lambda_list = [1e-7, 1e-7, 1e-7]

# Do SVM with Gaussian Kernel predictions

results0 = np.zeros(3000)
len_files = len(FILES)

for i in range(len_files):

    γ = gamma_list[i]
    λ = lambda_list[i]

    X_train, Y_train, X_test = load_data(i, data_dir=DATA_DIR, files_dict=FILES)

    kernel = GaussianKernel(γ)
    clf = SVM(_lambda=λ, kernel=kernel)
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    results0[i*1000:i*1000 + 1000] = y_pred

# SAVE Results
save_results("results_SVM_gaussian.csv", results0, RESULT_DIR)
print("1/3 Ending SVM with Gaussian kernel...")

#####################################
# 2) SVM with Convolutional kernel  #
#####################################
print("2/3 Starting SVM with Convolutional kernel...")
# Define parameters lists
sigma_list = [0.31,0.31,0.3]
k_list = [9,10,11]
lambda_list = [1e-5, 1e-9, 1e-9]

# Do SVM with Convolutional Kernel predictions
results1 = np.zeros(3000)

for i in range(len_files):

    sigma = sigma_list[i]
    k = k_list[i]
    λ = lambda_list[i]

    X_train, Y_train, X_test = load_data(i, data_dir=DATA_DIR, files_dict=FILES, mat=False)

    kernel = ConvKernel(sigma, k)
    clf = SVM(_lambda=λ, kernel=kernel)
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    results1[i*1000:i*1000 + 1000] = y_pred

# SAVE Results
save_results("results_conv_kernel.csv", results1, RESULT_DIR)
print("2/3 Ending SVM with Convolutional kernel...")

###########
# 3) CKN  #
###########
print("3/3 Starting CKN...")
# Define parameters lists
# for embeddings
k_list = [9, 9, 7]
sigma_list = [0.34, 0.3, 0.3]
n_anchors = 6000
β = 1e-3
# for SVM with Gaussian kernel
gamma_list = [184.375, 481.25, 560.41667]
lambda_list = [1e-12,1e-12,1e-12]
# choose random seed
np.random.seed(1702)

# Compute embeddings and do SVM with
# Do SVM with Convolutional Kernel predictions
results2 = np.zeros(3000)

for q in range(len_files):

    k = k_list[q]
    σ = sigma_list[q]
    γ = gamma_list[q]
    λ = lambda_list[q]

    # Choose random anchors
    kmers = compute_kmers_list(q, k)
    index = np.random.choice(range(len(kmers)), replace=False, size = n_anchors)
    anchors = kmers[index]

    # compute (K_ZZ + β Id)**(-0.5)
    K_ZZ_inv_sqr = K_zz_inv_sqr(anchors, σ, β)

    # compute embeddings
    X_train, Y_train, X_test = load_data(q, data_dir=DATA_DIR, files_dict=FILES, mat=False)
    E_train = compute_embeddings(X_train, np.array(anchors), k, σ, K_ZZ_inv_sqr)
    E_test = compute_embeddings(X_test, np.array(anchors), k, σ, K_ZZ_inv_sqr)

    # Do SVM Gaussian Kernel predictions on embeddings
    kernel = GaussianKernel(γ)
    clf = SVM(_lambda=λ, kernel=kernel)
    clf.fit(E_train, Y_train)
    y_pred = clf.predict(E_test)
    results2[q*1000:q*1000 + 1000] = y_pred

# SAVE Results
save_results("results_CKN.csv", results2, RESULT_DIR)
print("3/3 Ending CKN...")

###########
# Bagging #
###########
print("Starting bagging...")
# define the list of predictions to do bagging
file_list = ['results_SVM_gaussian.csv', 'results_conv_kernel.csv', 'results_CKN.csv']
savename = "Yte.csv"
y_pred = bagging(file_list, savename)
print("End bagging...")
