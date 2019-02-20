"""

Data utility functions, including:

- load_data
- save_results


"""
###########
# Imports #
###########

import csv
import numpy as np
import os

#############
# Utilities #
#############

def load_data(file_id, data_dir, files_dict, mat=True):
    """
    Load datasets
    
    Parameters
    ------------
    - file_id : int
        identifier to load the file
        (0, 1 or 2)
    
    - files_dict : dict
        Mapping from id to file names
    
    - data_dir : string
        Data folder path
    
    - mat : boolean (optional)
        If True, load the data in their matrix form
        If False, load the raw data sequences
        Default: True
        
    Returns
    ---------------
    - X_train : numpy.array
        Training data (either in matrix or string form)
    
    - Y_train : numpy.array
        Training label
        
    - X_test : numpy.array
        Testing data (either in matrix or string form)
    """
    
    X_train = list()
    Y_train = list()
    X_test = list()
    
    dic = files_dict[file_id]
    
    if mat:
        files = [dic["train_mat"], dic["label"], dic["test_mat"]]
    else:
        files = [dic["train"], dic["label"], dic["test"]]

    for file, l in zip(files, [X_train, Y_train, X_test]):
        with open(os.path.join(data_dir, file), "r", newline="") as csvfile:
            if "mat" in file:
                reader = csv.reader(csvfile, delimiter=" ")
                for row in reader:
                    l.append(row)
            else:
                reader = csv.reader(csvfile, delimiter=",")
                next(reader, None) # Skip the header
                for row in reader:
                    l.append(row[1])
                
    if mat:
        X_train = np.array(X_train).astype("float")
        Y_train = np.array(Y_train).astype("int")
        X_test = np.array(X_test).astype("float")
        np.random.seed(0)
        index = np.random.permutation(len(X_train))
        X_train = X_train[index]
        Y_train = Y_train[index]
    
    else:
        np.random.seed(0)
        index = np.random.permutation(len(X_train))
        X_train = [X_train[i] for i in index]
        Y_train = [Y_train[i] for i in index]
        Y_train = np.array(Y_train).astype("int")
        
    
    return X_train, Y_train, X_test



def save_results(filename, results, result_dir):
    """
    Save results in a csv file 
    
    Parameters
    -----------
    - filename : string
        Name of the file to be saved under the ``results`` folder
        
    - results : numpy.array
        Resulting array (0 and 1's)
        
    - result_dir : string
        Result folder path
    """
    
    assert filename.endswith(".csv"), "this is not a csv extension!"
    # Convert results to int
    results = results.astype("int")
    
    with open(os.path.join(result_dir, filename), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        # Write header
        writer.writerow(["Id", "Bound"]) 
        assert len(results) == 3000, "There is not 3000 predictions"
        # Write results
        for i in range(len(results)):
                writer.writerow([i, results[i]])