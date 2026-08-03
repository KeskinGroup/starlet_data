This folder includes gas-specific machine learning models that developed for CO₂, CH₄, N₂, and H₂ using molecular simulation data obtained for a large and diverse set of MOFs.

Three distinct ML approaches were considered:
(1) Developing ML models to predict gas uptakes and self-diffusivities of MOFs.
(2) Developing ML models to directly predict MOF gas permeabilities.
(3) Developing ML models to directly predict MOF/polymer MMMs gas permeabilities

The TPOT framework was used to optimize the ML pipelines. The number of generations defines how long the genetic algorithm evolves the model pipelines, while the population size specifies how many candidate pipelines are evaluated in each generation. A 5-fold cross-validation scheme was employed during training to assess model performance. The verbosity level was set to 2 to enable detailed reporting of the optimization process, and a fixed random seed was used to ensure reproducibility. The dataset was split into 80% training and 20% test sets, with the training set used for model development and the test set reserved for performance evaluation.

Using these three approaches, CO₂/CH₄, CO₂/N₂, and H₂/CO₂ membrane separations were systematically evaluated, enabling comparison of the predictive capabilities of each strategy and identification of promising MOF/polymer MMMs for each separation.

If you utilize the data from this repository, kindly cite it as: https://doi.org/10.1038/s43246-026-01207-9
