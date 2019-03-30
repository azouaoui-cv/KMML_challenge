import csv
import os
import numpy as np
from utils import save_results
from utils import FILES, DATA_DIR, RESULT_DIR



# define the list of predictions to do bagging
file_list = ['results_CKN.csv','results_conv_kernel.csv','results5.csv']


# load all the predictions vectors in a matrix
predictions_matrix = []
for file in file_list:
    l = []
    with open(os.path.join(RESULT_DIR, file), "r", newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            l.append(row[1])
    predictions_matrix.append(l[1:])
predictions_matrix = np.array(predictions_matrix).astype('int')


 # Do voting
y_pred = []
for i in range(len(l)-1):
    mean_vote = np.sum(predictions_matrix[:,i])/len(file_list)
    if mean_vote > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

# SAVE Results
save_results("results_bagging.csv", np.array(y_pred), RESULT_DIR)
