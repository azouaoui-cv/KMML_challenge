"""
Perform a batch greedy approach to use Nyström approximation

Made faster by Schur complement
"""



###########
# Imports #
###########
import os
import numpy as np
from tqdm import tqdm
from scipy import linalg
from utils import FILES, LOGGING_DIR
from utils.data import load_data, K1, compute_kmers_list
import logging
import argparse

#########
# Setup #
#########
DEFAULT_N_BATCH = 20
DEFAULT_N_ANCHORS_PER_BATCH = 300
# Logging filename
DEFAULT_LOGGING_FILENAME = "greedy.log"

sigma_dic = {0: 0.34, 1: 0.3, 2: 0.3}
k_dic = {0: 9, 1: 9, 2: 7}
# number of anchors for one batch

# choose initial random anchor
np.random.seed(1702)

# Create anchors_index folder
ANCHORS_INDEX_DIR = os.path.join(os.getcwd(), "anchors_index")
if not os.path.exists(ANCHORS_INDEX_DIR):
    os.mkdir(ANCHORS_INDEX_DIR)
    
# Argument parser
parser = argparse.ArgumentParser("Greedy script")
parser.add_argument("--batch",
                   help=f"Number of batch.  Default: {DEFAULT_N_BATCH}", type=int,
                   default=DEFAULT_N_BATCH)
parser.add_argument("--n-anchors-per-batch",
                   help=f"Number of anchors per batch.  Default: {DEFAULT_N_ANCHORS_PER_BATCH}", type=int,
                   default=DEFAULT_N_ANCHORS_PER_BATCH)
# Logging output filename
parser.add_argument("--logging-filename",
                   help=f"Filename of the logging file. Default: {DEFAULT_LOGGING_FILENAME}",
                   default=DEFAULT_LOGGING_FILENAME, type=str)


    
########
# Main #
########
if __name__ == "__main__":
    # Parser
    args = parser.parse_args()
    # Retrieve parser arguments
    n_batch = args.batch
    objectif = args.n_anchors_per_batch
    
    
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
    
    # Loop over files
    for idx in range(len(FILES)):
        
        σ = sigma_dic[idx]
        k = k_dic[idx]

        kmers = compute_kmers_list(idx, k)
        all_index = list()

        # Loop over batches
        for _ in range(n_batch):
            
            logger.info(f"batch: {_ + 1} / {n_batch} (dataset {idx})")

            # Randomly select first anchor
            n0 = np.random.randint(len(kmers))
            anchors = list(kmers[n0:n0+1])

            # init parameters
            n_anchors = len(anchors)
            index_list = [n0]

            # compute K_zz
            p = len(anchors)
            Z = np.array(anchors)
            K_zz = np.zeros((p,p))
            for j in range(p):
                for i in range(j+1):
                    K_zz[i,j] = K1(Z[i],Z[j], σ)
            K_zz =  K_zz + K_zz.T
            np.fill_diagonal(K_zz, np.diagonal(K_zz)/2)
            # compute K_zz inv
            K_ZZ_inv = linalg.inv(K_zz)

            # normalize kmers rows
            norms = np.linalg.norm(kmers, axis=1)
            kmers_normalize = kmers / norms.reshape(-1,1)
            norms2 = norms ** 2

            # Loop over anchors to be selected
            for m in range(objectif-1):

                # Select new anchor
                Z = np.array(anchors)
                Z = Z / np.linalg.norm(Z,axis=1).reshape(-1,1) # normalize Z rows
                S = Z.dot(kmers_normalize.T)
                S = np.exp((S - 1) / σ**2)
                S = np.einsum('i, ij -> ij',np.linalg.norm(Z,axis=1), S)
                S = np.einsum('j, ij -> ij',np.linalg.norm(kmers,axis=1), S)
                b = K_ZZ_inv.dot(S)
                b = np.einsum('ij,ij -> j', S, b)

                argmax = np.argmax(norms2 - b)
                new_anchor = kmers[argmax]

                # update the set of anchors
                anchors.append(new_anchor)
                index_list.append(argmax)

                # compute new K_zz_inv using Schur complement
                f = Z.dot(new_anchor / np.linalg.norm(new_anchor))
                f = np.exp((f - 1) / σ**2).reshape(-1, 1)
                f = np.einsum('i, ij -> ij', np.linalg.norm(Z, axis=1), f)
                f = np.einsum('j, ij -> ij', np.linalg.norm(new_anchor.reshape(1, -1), axis=1), f)
                b = K_ZZ_inv.dot(f)
                s = np.linalg.norm(new_anchor) ** 2 - f.T.dot(K_ZZ_inv.dot(f))
                A = K_ZZ_inv + (1 / s) * b.dot(b.T)
                B = - (1 / s) * b
                C = - (1 / s) * b.T
                D = 1 / s
                AB = np.concatenate((A,B), axis=1)
                CD = np.concatenate((C,D), axis=1)
                K_ZZ_inv = np.concatenate((AB,CD), axis=0)

            # Update index list
            all_index += index_list

        # Save index list
        np.save(os.path.join(ANCHORS_INDEX_DIR, f'index_list_d{idx}_k{k}_s{round(σ, 2)}_b{n_batch}.npy'), 
                np.array(all_index))
