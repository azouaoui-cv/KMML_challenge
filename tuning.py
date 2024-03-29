"""

This file is used for tuning our existing models
using the parameters range provided
on either raw data or a pre-computed array representation 

"""
###########
# Imports #
###########
import logging
import numpy as np
from utils import load_data
from utils import cross_validation
from kernels import LinearKernel, GaussianKernel, ConvKernel
from utils import FILES, DATA_DIR, RESULT_DIR, LOGGING_DIR
from models import SVM, SPR, SVM_precomputed_gram
import argparse
import os
from itertools import product


#########
# Paths #
#########
if not os.path.isdir(LOGGING_DIR):
    os.mkdir(LOGGING_DIR)


#########
# Setup #
#########
# Choice values"
CHOICES_MODEL = ["SVM", "SPR", "SVM_precomputed_gram"]
CHOICES_KERNEL = ["Gaussian", "Linear", "Conv"]

# Default values
DEFAULT_MODEL = "SVM"
DEFAULT_KERNEL = "Gaussian"
# Lambda
DEFAULT_LAMBDA_MIN = -7
DEFAULT_LAMBDA_MAX = -4
DEFAULT_LAMBDA_NUM = 10
DEFAULT_LAMBDA_LOGSCALE = True
DEFAULT_USE_LAMBDA = False
# Gamma
DEFAULT_GAMMA_MIN = 50
DEFAULT_GAMMA_MAX = 600
DEFAULT_GAMMA_NUM = 51
DEFAULT_GAMMA_LOGSCALE = False
DEFAULT_USE_GAMMA = False
# Sigma
DEFAULT_SIGMA_MIN = 0.26
DEFAULT_SIGMA_MAX = 0.36
DEFAULT_SIGMA_NUM = 3
DEFAULT_USE_SIGMA = False
# Window size
DEFAULT_WINDOW_SIZE_MIN = 8
DEFAULT_WINDOW_SIZE_MAX = 10
DEFAULT_WINDOW_SIZE_NUM = 3
DEFAULT_USE_WINDOW_SIZE = False
# Numerical representation
DEFAULT_USE_MAT = False
# Number of cross-validation folds
DEFAULT_K_FOLD = 5
# Logging filename
DEFAULT_LOGGING_FILENAME = "SVM_tuning.log"
# precompute gram matrix
DEFAULT_PRECOMPUTE_GRAM = False



##########################
# Command line arguments #
##########################
import argparse

parser = argparse.ArgumentParser("Tuning script")

# Model
parser.add_argument("-c", "--clf",
                    help=f"Classifier name.  Default: {DEFAULT_MODEL}.  Choices: {CHOICES_MODEL}",
                   default=DEFAULT_MODEL, choices=CHOICES_MODEL, type=str)
# Kernel
parser.add_argument("-k", "--kernel",
                    help=f"Kernel name.  Default: {DEFAULT_KERNEL}.  Choices: {CHOICES_KERNEL}",
                   default=DEFAULT_KERNEL, choices=CHOICES_KERNEL, type=str)
# Lambda
parser.add_argument("--lambda-min",
                   help=f"Regularizer lowest value.  Default: {DEFAULT_LAMBDA_MIN}", type=float,
                   default=DEFAULT_LAMBDA_MIN)
parser.add_argument("--lambda-max",
                   help=f"Regularizer highest value.  Default: {DEFAULT_LAMBDA_MAX}", type=float,
                   default=DEFAULT_LAMBDA_MAX)
parser.add_argument("--lambda-num",
                   help=f"Number of lambda values to try.  Default: {DEFAULT_LAMBDA_NUM}", type=int,
                   default=DEFAULT_LAMBDA_NUM)
parser.add_argument("--lambda-logscale",
                   help=f"Use logscale for regularizer tuning.  Default: {DEFAULT_LAMBDA_LOGSCALE}", type=bool,
                   default=DEFAULT_LAMBDA_LOGSCALE)
parser.add_argument("--use-lambda",
                   help=f"Whether to use the regularizer. Default: {DEFAULT_USE_LAMBDA}",
                   default=DEFAULT_USE_LAMBDA, action="store_true")
# Gamma
parser.add_argument("--gamma-min",
                   help=f"Scaling lowest value.  Default: {DEFAULT_GAMMA_MIN}", type=float,
                   default=DEFAULT_GAMMA_MIN)
parser.add_argument("--gamma-max",
                   help=f"Scaling highest value.  Default: {DEFAULT_GAMMA_MAX}", type=float,
                   default=DEFAULT_GAMMA_MAX)
parser.add_argument("--gamma-num",
                   help=f"Number of Gamma values to try.  Default: {DEFAULT_GAMMA_NUM}", type=int,
                   default=DEFAULT_GAMMA_NUM)
parser.add_argument("--gamma-logscale",
                   help=f"Use logscale for Gamma tuning.  Default: {DEFAULT_GAMMA_LOGSCALE}", type=bool,
                   default=DEFAULT_GAMMA_LOGSCALE)
parser.add_argument("--use-gamma",
                   help=f"Whether to use the scaling. Default: {DEFAULT_USE_GAMMA}",
                   default=DEFAULT_USE_GAMMA, action="store_true")
# Sigma
parser.add_argument("--sigma-min",
                   help=f"Regularizer lowest value.  Default: {DEFAULT_SIGMA_MIN}", type=float,
                   default=DEFAULT_SIGMA_MIN)
parser.add_argument("--sigma-max",
                   help=f"Regularizer highest value.  Default: {DEFAULT_SIGMA_MAX}", type=float,
                   default=DEFAULT_SIGMA_MAX)
parser.add_argument("--sigma-num",
                   help=f"Number of sigma values to try.  Default: {DEFAULT_SIGMA_NUM}", type=int,
                   default=DEFAULT_SIGMA_NUM)
parser.add_argument("--use-sigma",
                   help=f"Whether to use the sigma in conv kernel. Default: {DEFAULT_USE_SIGMA}",
                   default=DEFAULT_USE_SIGMA, action="store_true")

# Window size
parser.add_argument("--window-size-min",
                   help=f"Window size lowest value.  Default: {DEFAULT_WINDOW_SIZE_MIN}", type=int,
                   default=DEFAULT_WINDOW_SIZE_MIN)
parser.add_argument("--window-size-max",
                   help=f"Window size highest value.  Default: {DEFAULT_WINDOW_SIZE_MAX}", type=int,
                   default=DEFAULT_WINDOW_SIZE_MAX)
parser.add_argument("--window-size-num",
                   help=f"Number of window size values to try.  Default: {DEFAULT_WINDOW_SIZE_NUM}", type=int,
                   default=DEFAULT_WINDOW_SIZE_NUM)
parser.add_argument("--use-window-size",
                   help=f"Whether to use the window size hyperparameter. Default: {DEFAULT_USE_WINDOW_SIZE}",
                   default=DEFAULT_USE_WINDOW_SIZE, action="store_true")


# Use numerical array representation (alternative: raw sequence)
parser.add_argument("--use-mat",
                   help=f"Whether to use the numerical array representation. Default: {DEFAULT_USE_MAT}",
                   default=DEFAULT_USE_MAT, action="store_true")
# Number of k folds in cross-validation
parser.add_argument("--k-fold",
                   help=f"Number of folds to use in cross-validation. Default: {DEFAULT_K_FOLD}",
                   default=DEFAULT_K_FOLD, type=int)
# Logging output filename
parser.add_argument("--logging-filename",
                   help=f"Filename of the logging file. Default: {DEFAULT_LOGGING_FILENAME}",
                   default=DEFAULT_LOGGING_FILENAME, type=str)
# precompute Gram matrix before cross-validation
parser.add_argument("--use-precompute-gram",
                   help=f"Wheter to use the precompute gram matrix before cross-validation. Default: {DEFAULT_PRECOMPUTE_GRAM}",
                   default=DEFAULT_PRECOMPUTE_GRAM, action="store_true")



########
# Main #
########
if __name__ == "__main__":
    
    # Parser options
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    datefmt='%m/%d/%Y %I:%M:%S %p',
    handlers=[
        logging.FileHandler(filename=os.path.join(LOGGING_DIR, args.logging_filename)),
        logging.StreamHandler()
    ])

    logger = logging.getLogger()
    
    ############
    # Settings #
    ############
    logger.info("Start")
    logger.info(f"args: {args}")

    kernel_name = args.kernel
    model_name = args.clf
    
    # Populate hyperparameters list
    if args.use_lambda:
        if args.lambda_logscale:
            lambda_list = np.logspace(args.lambda_min, args.lambda_max, args.lambda_num, endpoint=True)
        else:
            lambda_list = np.linspace(args.lambda_min, args.lambda_max, args.lambda_num, endpoint=True)
    else:
        lambda_list = [0]

    if args.use_gamma:
        if args.gamma_logscale:
            gamma_list = np.logspace(args.gamma_min, args.gamma_max, args.gamma_num, endpoint=True)
        else:
            gamma_list = np.linspace(args.gamma_min, args.gamma_max, args.gamma_num, endpoint=True)
    else:
        gamma_list = [0]

    if args.use_sigma:
        sigma_list = np.linspace(args.sigma_min, args.sigma_max, args.sigma_num, endpoint=True)
    else:
        sigma_list = [0]

    if args.use_window_size:
        window_size_list = np.linspace(args.window_size_min, args.window_size_max, args.window_size_num, endpoint=True)
    else:
        window_size_list = [0]

    # Adjust settings accordingly
    if args.use_precompute_gram:
        settings = list(product(sigma_list, window_size_list))
    else:
        settings = list(product(gamma_list, lambda_list, sigma_list, window_size_list))

    len_files = len(FILES)

    best_score = {i: 0 for i in range(len_files)}
    best_lambda = {i: 0 for i in range(len_files)}
    best_gamma = {i: 0 for i in range(len_files)}
    best_sigma = {i: 0 for i in range(len_files)}
    best_window_size = {i: 0 for i in range(len_files)}
    
    N = len(settings)

    for _, params in enumerate(settings):

        logger.info(f"Run {_ + 1} / {N}")
        
        if args.use_precompute_gram:
            gamma, _lambda = None, None
            sigma, window_size = params
        else:
            gamma, _lambda, sigma, window_size = params
        # convert window_size to int
        window_size = int(window_size)

        if kernel_name == "Gaussian":
            kernel = GaussianKernel(gamma)

        elif kernel_name == "Linear":
            kernel = LinearKernel()

        elif kernel_name == "Conv":
            kernel = ConvKernel(sigma=sigma, k=window_size)

        if model_name == "SVM":
            clf = SVM(_lambda=_lambda, kernel=kernel)

        elif model_name == "SPR":
            clf = SPR(kernel=kernel)

        elif model_name =="SVM_precomputed_gram":
            if args.use_precompute_gram:
                clf = None # To be defined later
            else:
                clf = SVM_precomputed_gram(_lambda=_lambda, kernel=kernel)


        for i in range(len_files):

            if args.use_precompute_gram:

                # Load data
                X_train, Y_train, X_test = load_data(i, data_dir=DATA_DIR, files_dict=FILES, mat=args.use_mat)

                # compute gram matrix of ALL dataset 
                K = kernel.compute_gram_matrix(X_train)
                
                # Same Gram matrix to be used on different lambdas
                for _lambda in lambda_list:
                    assert model_name == "SVM_precomputed_gram"
                    clf = SVM_precomputed_gram(_lambda=_lambda, kernel=kernel)

                    # cross validation (default: k=5)
                    results = cross_validation(i, clf, k=args.k_fold, data_dir=DATA_DIR, files_dict=FILES, mat=args.use_mat, K=K)
                
                    score_train = results["train_avg"]
                    score_val = results["val_avg"]
                    logger.info(f"Accuracy on train set / val set {i} : {round(score_train, 3)} / {round(score_val, 3)}"
                             f"(lambda: {_lambda}, gamma: {gamma}, sigma: {sigma}, window_size: {window_size})")
                    
                    if score_val > best_score[i]:
                        best_score[i] = score_val
                        best_lambda[i] = _lambda
                        best_gamma[i] = gamma
                        best_sigma[i] = sigma
                        best_window_size[i] = window_size

                        logger.info("New best on {0}".format(i))
                
                
            else:
                results = cross_validation(i, clf, k=args.k_fold, data_dir=DATA_DIR, files_dict=FILES, mat=args.use_mat)
                score_train = results["train_avg"]
                score_val = results["val_avg"]
                logger.info(f"Accuracy on train set / val set {i} : {round(score_train, 3)} / {round(score_val, 3)}"
                             f"(lambda: {_lambda}, gamma: {gamma}, sigma: {sigma}, window_size: {window_size})")


                if score_val > best_score[i]:
                    best_score[i] = score_val
                    best_lambda[i] = _lambda
                    best_gamma[i] = gamma
                    best_sigma[i] = sigma
                    best_window_size[i] = window_size

                    logger.info("New best on {0}".format(i))

    # Save best configuration
    logger.info(f"Best score: {best_score}")
    if args.use_gamma:
        logger.info(f"Best gamma: {best_gamma}")
    if args.use_lambda:
        logger.info(f"Best lambda: {best_lambda}")
    if args.use_sigma:
        logger.info(f"Best sigma: {best_sigma}")
    if args.use_window_size:
        logger.info(f"Best window size: {best_window_size}")

    logger.info("End")