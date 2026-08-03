This is the folder of models developed for the ML study of MOFs for membrane-based CO<sub>2</sub>/N<sub>2</sub>, CO<sub>2</sub>/CH<sub>4</sub>, N<sub>2</sub>/CH<sub>4</sub>, H<sub>2</sub>/CO<sub>2</sub>, H<sub>2</sub>/CH<sub>4</sub>, H<sub>2</sub>/N<sub>2</sub>, and O<sub>2</sub>/N<sub>2</sub> separations.

•	.xlsx files contain the input features for MOFs used to build the ML models. The columns represent various structural and guest-related features of MOFs, and the last column is the target data (e.g., CO<sub>2</sub> diffusivity at 1 bar).

•	TPOT input parameters: generation parameter means the genetic algorithm will run for the specified number of generations, evolving the model pipeline over time. The population size indicates that the specified number of different model pipelines will be evaluated in each generation. Cross-validation (cv) is set to 5, meaning a 5-fold cross-validation will be used to assess the model performance during training. The verbosity level is set to 2, providing detailed logging of the training process. A random number ensures the reproducibility of results. The data is split into 80% training set and 20% test set, where 80% of the data is used for training the models and the remaining 20% for evaluating model performance.

•	.py files contain the best ML pipelines that are identified for predicting the gas diffusivity data at 1 bar.

•	The publicly available web interface is available on https://mofgasdiffusionpredictor.streamlit.app/ for predicting gas diffusivities of MOFs to facilitate material selection and enable broader applicability in materials discovery.
