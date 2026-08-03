This is the repository that includes the ML models developed to predict drug storage capacities of MOFs. 

- .xlsx files contain the input features for MOFs used to build the ML models. The columns represent various structural features of MOFs and DFT-based features of drug molecules, and the last column is the target data (e.g., 5-FU uptakes at 1 bar).

-	TPOT input parameters: generation parameter is set to 10, meaning the genetic algorithm will run for 10 generations, evolving the model pipeline over time. The population size of 30 indicates that 30 different model pipelines will be evaluated in each generation. Cross-validation (cv) is set to 5, meaning a 5-fold cross-validation will be used to assess the model performance during training. The verbosity level is set to 2, providing detailed logging of the training process. A random number seed of 42 ensures the reproducibility of results. The data is split into 80% training set and 20% test set, where 80% of the data is used for training the models and the remaining 20% for evaluating model performance.

- .py files contain the best ML pipelines that are identified for predicting the drug adsorption data at 1 bar.

- This repository has been archived on Zenodo. The archived version associated with the manuscript is available at: https://doi.org/10.5281/zenodo.21340484

If you utilize the data from this repository, kindly cite it as: https://doi.org/10.1039/d6dd00223d
