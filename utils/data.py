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


def cross_validation(dataset_idx, clf, data_dir, files_dict,
                     k=5, embeddings=None, embeddings_path=None):
    """
    Perform a k-fold cross-validation on a specific dataset
    given a specific classifier
    
    Parameters
    -------------
    - dataset_idx : int
        Dataset index to be called in the data loader
        
    - clf : object
        Classifier object with methods:
        . fit
        . predict
        . score
        
    - files_dict : dict
        Mapping from id to file names
    
    - data_dir : string
        Data folder path
        
    - k : int (optional)
        Number of folds
        Default: 5
        
    - embeddings_path : str (optional)
        pre-computed embeddings filename
    
    - embeddings : np.ndarray (optional)
        Computed embeddings available in memory
        
    
    Returns
    -----------
    - results : dictionary
        Summary of the results
        Note: the results can be display using
        a pandas.DataFrame such as in:
        ``pd.DataFrame(results)``
    """
    
    # Setup
    scores_val = list()
    scores_train = list()
    
    # Load data
    X_train, Y_train, X_test = load_data(dataset_idx, data_dir=data_dir, files_dict=files_dict)
    
    if embeddings_path is not None:
        X_train = np.load(embeddings_path)
        
    if embeddings is not None:
        X_train = embeddings
    
    n = len(X_train)
    assert n == len(Y_train)
    
    # Divise the samples
    bounds = [(i * (n // k), (i+1) * (n // k))
              for i in range(k)]


    # Loop through the divided samples
    for bound in bounds:
        lower, upper = bound
        # Create index array for validation set
        idx = np.arange(lower, upper)
        not_idx = [i for i in range(n) if i not in idx]

        # Populate current train and val sets
        _X_val = X_train[idx]
        _Y_val = Y_train[idx]
        _X_train = X_train[not_idx]
        _Y_train = Y_train[not_idx]

        # Sanity checks
        assert len(_X_train) == len(_Y_train)
        assert len(_X_val) == len(_Y_val)
        assert len(_X_train) == n - len(X_train) // k

        # Fit the classifier on the current training set
        clf.fit(_X_train, _Y_train)
        # Compute the score
        y_pred_train = clf.predict(_X_train)
        y_pred_val = clf.predict(_X_val)
        score_train = clf.score(y_pred_train, _Y_train)
        score_val = clf.score(y_pred_val, _Y_val)

        scores_val.append(score_val)
        scores_train.append(score_train)


   
    # Format the results in a dictionary
    # Compute the score average and standard deviation
    results = {"train_scores": scores_train,
               "val_scores": scores_val,
               "train_avg": np.mean(scores_train),
               "val_avg": np.mean(scores_val),
               "train_std": np.std(scores_train),
               "val_std": np.std(scores_val)}
    
    return results



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


def compute_kmers_list(idx, k):


    """This function compute all the k-mers of a list of sequences

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
