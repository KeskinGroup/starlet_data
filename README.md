This repository contains the machine learning (ML) models developed in this work to predict single-component gas adsorption performances of triazine-linked ReDD-hypoCOFs at 1 bar. The ML models were used to efficiently screen large subsets of the ReDD-COFFEE database and identify top-performing materials for gas adsorption and separation applications. 

Machine learning models were developed using the Tree-based Pipeline Optimization Tool (TPOT) (version 0.12.2) to automatically identify optimal regression pipelines and perform hyperparameter optimization. Regression algorithms from the scikit-learn library were employed during model selection. TPOT input parameters: generation parameter is set to 10, meaning the genetic algorithm will run for 10 generations, evolving the model pipeline over time. The population size of 50 indicates that 50 different model pipelines will be evaluated in each generation. Cross-validation (cv) is set to 5, meaning a 5-fold cross-validation will be used to assess the model performance during training. A random number seed of 42 ensures the reproducibility of results. 

To ensure a consistent distribution of the feature space between training and test sets, a stratified sampling strategy was applied. The datasets were split into 80% training and 20% testing subsets.

.csv files
These files contain the full datasets used to train and test the ML models. Each file includes structural, chemical, and energetic descriptors of triazine ReDD-hypoCOFs, together with the corresponding target adsorption values (CO<sub>2</sub>, CH<sub>4</sub>, O<sub>2</sub>, N<sub>2</sub>, and H<sub>2</sub> uptakes at 1 bar).

.py files
These scripts contain the optimized ML pipelines obtained in this study. Each Python file corresponds to the best-performing ML model for predicting the adsorption uptake of a specific gas at 1 bar.

If you utilize the data from this repository, kindly cite it as: https://doi.org/10.1021/acs.iecr.5c04806
