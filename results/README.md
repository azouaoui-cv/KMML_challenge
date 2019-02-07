# Results

0. "results0.csv": assign the 0 label to all the test data. Score (on public test dataset): 0.51266
1. "results1.csv": implements the Simple Pattern Recognition algorithm separately to each train/test couple ($k = (0, 1, 2)$). Score: 0.52133 (Big mistake : error in data loading)
2. "results2.csv": SPR linear kernel after data loader refactoring. Score: 0.59200 
2. "results3.csv": SPR linear kernel on whole set. Score: 0.57599
