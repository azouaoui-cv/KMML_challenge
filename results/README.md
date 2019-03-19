# Results

0. "results0.csv": assign the 0 label to all the test data. Score (on public test dataset): 0.51266
1. "results1.csv": implements the Simple Pattern Recognition algorithm separately to each train/test couple ($k = (0, 1, 2)$). Score: 0.52133 (Big mistake : error in data loading)
2. "results2.csv": SPR linear kernel after data loader refactoring. Score: 0.59200 
3. "results3.csv": SPR linear kernel on whole set. Score: 0.57599
4. "results4.csv": SVM Gaussian kernel same parameters for each dataset. Score: 0.68066 
5. "results5.csv": SPR Gaussian kernel different parameters for each dataset. Score: 0.67600
6. "results_conv_kernel.csv": Convolutional Kernel not approximate. Score: 0.70133
