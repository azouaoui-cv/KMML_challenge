"""
Perform cross-validation hyperparameters tuning
on pre-computed embeddings
"""
###########
# Imports #
###########
import logging
import numpy as np
from utils.data import load_data
from utils.data import cross_validation
from utils.data import filename_parser
from utils.kernels import LinearKernel, GaussianKernel, ConvKernel
from utils import FILES, DATA_DIR, RESULT_DIR, LOGGING_DIR, EMBEDDING_DIR
from utils.models import SVM, SPR
import argparse
import os
from itertools import product
import argparse

###########
# GLOBALS #
###########

# Choice values
CHOICES_MODEL = ["SVM", "SPR"]
CHOICES_KERNEL = ["Gaussian", "Linear"]

# Default values
DEFAULT_MODEL = "SVM"
DEFAULT_KERNEL = "Gaussian"
# Lambda
DEFAULT_LAMBDA_MIN = -9
DEFAULT_LAMBDA_MAX = -5
DEFAULT_LAMBDA_NUM = 2
DEFAULT_LAMBDA_LOGSCALE = True
# Gamma
DEFAULT_GAMMA_MIN = 100
DEFAULT_GAMMA_MAX = 300
DEFAULT_GAMMA_NUM = 2
DEFAULT_GAMMA_LOGSCALE = False
# Number of cross-validation folds
DEFAULT_K_FOLD = 5
# Logging filename
DEFAULT_LOGGING_FILENAME = "test.log"

##############################
### Command line arguments ###
##############################



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
# Number of k folds in cross-validation
parser.add_argument("--k-fold",
                   help=f"Number of folds to use in cross-validation. Default: {DEFAULT_K_FOLD}",
                   default=DEFAULT_K_FOLD, type=int)
# Logging output filename
parser.add_argument("--logging-filename",
                   help=f"Filename of the logging file. Default: {DEFAULT_LOGGING_FILENAME}",
                   default=DEFAULT_LOGGING_FILENAME, type=str)



########
# MAIN #
########

if __name__ == "__main__":
    
    # Argument Parser options
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
    
    logger.info("Start")
    logger.info(f"args: {args}")
    
    # Model and kernel selection
    kernel_name = args.kernel
    model_name = args.clf
        
    # Populate hyperparameters list  
    # Lambda
    if args.lambda_logscale:
        lambda_list = np.logspace(args.lambda_min, args.lambda_max, args.lambda_num, endpoint=True)
    else:
        lambda_list = np.linspace(args.lambda_min, args.lambda_max, args.lambda_num, endpoint=True)   
    # Gamma
    if args.gamma_logscale:
        gamma_list = np.logspace(args.gamma_min, args.gamma_max, args.gamma_num, endpoint=True)
    else:
        gamma_list = np.linspace(args.gamma_min, args.gamma_max, args.gamma_num, endpoint=True)  

    # Grid Search setup    
    settings = list(product(gamma_list, lambda_list))
    
    len_files = len(FILES)
    
    best_score = {i: 0 for i in range(len_files)}
    best_lambda = {i: 0 for i in range(len_files)}
    best_gamma = {i: 0 for i in range(len_files)}
    best_sigma = {i: 0 for i in range(len_files)}
    best_window_size = {i: 0 for i in range(len_files)}
    
    
    # Main loop
    for _, params in enumerate(settings):
        
        gamma, _lambda, = params
        
        if kernel_name == "Gaussian":
            kernel = GaussianKernel(gamma)

        elif kernel_name == "Linear":
            kernel = LinearKernel()

        if model_name == "SVM":
            clf = SVM(_lambda=_lambda, kernel=kernel)

        elif model_name == "SPR":
            clf = SPR(kernel=kernel)

        # Loop from pre-computed embeddings
        #for filename in os.listdir(EMBEDDING_DIR)[:1]: # small test
        for filename in os.listdir(EMBEDDING_DIR):
        
            # Full path
            file_path = os.path.join(EMBEDDING_DIR, filename)
            # Parsing
            dataset_idx, sigma, window_size = filename_parser(filename)
            # Cross validation
            results = cross_validation(dataset_idx=dataset_idx, clf=clf,
                                       data_dir=DATA_DIR, files_dict=FILES,
                                       k=5, embeddings_path=file_path, mat=True)
            # Process scores
            score_train = results["train_avg"]
            score_val = results["val_avg"]
            logger.info(f"Accuracy on train set / val set {dataset_idx} : {round(score_train, 3)} / {round(score_val, 3)}"
                         f"(λ: {_lambda},γ: {gamma}, sigma: {sigma}, window_size: {window_size})")
            # Update best
            if score_val > best_score[dataset_idx]:
                best_score[dataset_idx] = score_val
                best_lambda[dataset_idx] = _lambda
                best_gamma[dataset_idx] = gamma
                best_sigma[dataset_idx] = sigma
                best_window_size[dataset_idx] = window_size

                logger.info("\n")

        # Save best configuration
        logger.info(f"Best score: {best_score}")
        logger.info(f"Best gamma: {best_gamma}")
        logger.info(f"Best lambda: {best_lambda}")
        logger.info(f"Best sigma: {best_sigma}")
        logger.info(f"Best window size: {best_window_size}")
        
        logger.info("End")
    