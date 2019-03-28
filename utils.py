"""

utility functions, including:

- load_data
- save_results
- filename_parser
- cross_validation
- bagging


"""

###########
# Imports #
###########

import csv
import numpy as np
import os

################
# Define paths #
################

FILES = {0: {"train_mat": "Xtr0_mat100.csv",
             "train": "Xtr0.csv",
             "test_mat": "Xte0_mat100.csv",
             "test": "Xte0.csv",
             "label": "Ytr0.csv"},
         1: {"train_mat": "Xtr1_mat100.csv",
             "train": "Xtr1.csv",
             "test_mat": "Xte1_mat100.csv",
             "test": "Xte1.csv",
             "label": "Ytr1.csv"},
         2: {"train_mat": "Xtr2_mat100.csv",
             "train": "Xtr2.csv",
             "test_mat": "Xte2_mat100.csv",
             "test": "Xte2.csv",
             "label": "Ytr2.csv"}}

CWD = os.getcwd()
DATA_DIR = os.path.join(CWD, "data")
RESULT_DIR = os.path.join(CWD, "results")
LOGGING_DIR = os.path.join(CWD, "logging")
EMBEDDING_DIR = os.path.join(CWD, "embeddings")



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


def filename_parser(filename):
    """
    Parse embeddings filename
    to extract relevant hyperparameters information

    Parameters
    ----------
    - filename : string
        Filename (not the entire path)

    Returns
    ---------
    - dataset_idx : int

    - sigma : float

    - window_size : int
    """
    parser = filename.split("_")
    dataset_idx = int(parser[1][1])
    sigma = float(parser[2][1:])
    window_size = int(parser[3][1:].split(".")[0])
    print("Loading file: {0}".format(filename))
    return dataset_idx, sigma, window_size


def cross_validation(dataset_idx, clf, data_dir, files_dict,
                     k=5, embeddings=None, embeddings_path=None, mat=False, K=None):
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

    - mat : boolean
        Whether to take the original pre-processed embeddings

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
    X_train, Y_train, X_test = load_data(dataset_idx, data_dir=data_dir, files_dict=files_dict, mat=mat)
    # Temporary lower data size
    #X_train = X_train[:100]
    #Y_train = Y_train[:100]

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
        # Assign bounds
        lower, upper = bound

        # Create index array for validation set
        idx = np.arange(lower, upper)
        not_idx = [i for i in range(n) if i not in idx]

        # Populate current train and val sets
        # Handle arrays
        if mat:
            _X_val = X_train[idx]
            _X_train = X_train[not_idx]
        # Handle lists
        else:
            _X_val = X_train[lower:upper]
            _X_train = X_train[:lower] + X_train[upper:]

        _Y_val = Y_train[idx]
        _Y_train = Y_train[not_idx]

        # Sanity checks
        assert len(_X_train) == len(_Y_train)
        assert len(_X_val) == len(_Y_val)
        assert len(_X_train) == n - len(X_train) // k

        # Compute the of the corresponding portion of the gram matrix if given
        if K is not None:

            G = np.take(np.take(K, not_idx, axis = 0), not_idx ,axis = 1)
            S = np.take(np.take(K, idx, axis = 0), not_idx ,axis = 1)

            # Fit the classifier on the current training set
            clf.fit(_X_train, _Y_train, G)
            # Compute the score
            y_pred_train = clf.predict(_X_train, G)
            y_pred_val = clf.predict(_X_val, S)
            score_train = clf.score(y_pred_train, _Y_train)
            score_val = clf.score(y_pred_val, _Y_val)

            scores_val.append(score_val)
            scores_train.append(score_train)
        else:
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


def bagging(file_list, savename):
    """
    Perform a bagging on the predictions given and save the result

    Parameters
    -------------
    - file_list : list of string
        List of the prediction filenames to do bagging

    - savename : string
        Name of the file to save the result of bagging

    Returns
    -----------
    - results : array
        bagging of the prediction given in input
    """

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
    save_results(savename, np.array(y_pred), RESULT_DIR)

    return y_pred
