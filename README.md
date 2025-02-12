This is the folder of models developed for the ML study of COFs for adsorption-based CO<sub>2</sub>/CH<sub>4</sub>, CO<sub>2</sub>/H<sub>2</sub>, CO<sub>2</sub>/N<sub>2</sub>, CH<sub>4</sub>/H<sub>2</sub>, CH<sub>4</sub>/N<sub>2</sub>, and O<sub>2</sub>/N<sub>2</sub> separations. 

- .csv files contain the input features for COFs used to build the ML models. The columns represent various structural, chemical and energetic features of COFs, and the last column is the target data (e.g., CO<sub>2</sub> adsorption at 0.1 bar).

-	TPOT input parameters: generation parameter is set to 10, meaning the genetic algorithm will run for 10 generations, evolving the model pipeline over time. The population size of 50 indicates that 50 different model pipelines will be evaluated in each generation. Cross-validation (cv) is set to 5, meaning a 5-fold cross-validation will be used to assess the model performance during training. The verbosity level is set to 2, providing detailed logging of the training process. A random number seed of 42 ensures the reproducibility of results. The data is split into 80% training set and 20% test set, where 80% of the data is used for training the models and the remaining 20% for evaluating model performance.

- .py files contain the best ML pipelines that are identified for predicting the gas adsorption data at various pressures.

If you utilize the data from this repository, kindly cite it as follows: https://doi.org/10.1021/acsmaterialslett.4c02594
