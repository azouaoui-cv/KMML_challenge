import csv
import numpy as np
import os
from utils.data import load_data, save_results
from utils.models import SVM
from utils.kernels import GaussianKernel
from utils import FILES



# DEFINE PARAMETERS
γ = 275
λ = 6e-6

# Do SVM + Gaussian Kernel predictions
kernel = GaussianKernel(γ)
results = np.zeros(3000)
len_files = len(FILES)

for i in range(len_files):

    X_train, Y_train, X_test = load_data(i, data_dir=DATA_DIR, files_dict=FILES)
    clf = SVM(_lambda=λ, kernel=kernel)
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    results[i*1000:i*1000 + 1000] = y_pred


# SAVE Results
save_results("results_SVM_gaussian.csv", results, RESULT_DIR)
