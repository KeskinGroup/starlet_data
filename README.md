This is the folder of models developed for the ML study of IL/ZIF-8 composites for adsorption-based CO2/CH4 and CO2/N2 separation.

.xlsx files contain the input features for IL/ZIF-8 composites used to build the ML models. The columns represent various structural and chemical features of the ILs, and the last column is the target data (e.g., CO2 adsorption at 0.1 bar).

TPOT input parameters.csv file contains the generation parameter is set to 10, meaning the genetic algorithm will run for 10 generations, evolving the model pipeline over time. The population size of 25 indicates that 25 different model pipelines will be evaluated in each generation. Cross-validation (cv) is set to 5, meaning a 5-fold cross-validation will be used to assess the model performance during training. The verbosity level is set to 2, providing detailed logging of the training process. A random number seed of 42 ensures the reproducibility of results. The data is split into 80% training set and 20% test set, where 80% of the data is used for training the models and the remaining 20% for evaluating model performance.

.py files contain the best ML pipelines that are identified for predicting the gas adsorption data at various pressures.
