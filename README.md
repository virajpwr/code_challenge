# Coding Challenge

## Data Preparation

### The following steps were taken to prepare the data:

1. Read raw data and target data.
2. Lowercase column names.
3. Rename column names: 'unnamed: 0' to index and 'unnamed: 17' to unnamed_17
4. Fill missing value of groups with the mode of the groups on date 21/01/2020.
5. Convert datatypes.
6. join raw data and target data on index for each group.
7. Rename and drop columns.

#### The flow chart below shows the steps taken to prepare the data.

![](flowchart/new/dataprep.jpg)

## Data preprocessing

### The following steps were taken to preprocess the data:

1. Drop duplicate column.
2. convert datatypes.
3. replace values for columns cycle, crystal_supergroup, etherium_before_start.
4. replace missing values of continous variables with the median of the variable and missing values of categorical variables with the mode of the variable.
5. Drop duplicate rows.
6. Remove outliers with IQR method.

#### The flow chart below shows the steps taken to preprocess the data.

![](flowchart/new/preprocessing.jpg)

## Feature Engineering

### The following steps were taken to engineer features:

1. Spit the 'when' column in day, weekday, month. Perform one hot encoding on the weekday column.
2. Calculate time difference between timestamps
3. label encoding for categorical columns. To deal with high cardinality, we will replace the categories with low frequency with a single category.
4. Target encoding done to deal with high cardinality in the categorical columns.
5. Count encoding on categorical columns.
6. Gaussian rank normalization on continous columns and target to normalize the data and reduce the effect of outliers.

#### The flow chart below shows the steps taken to engineer features.

![](flowchart/feature%20engineering.jpeg)

## Feature selection

### The following steps were taken to select features:

#### Continous variable:

1. Calculate Mutual information score between continous variables and target.
2. Take non zero mutual information score variables.
3. Calculate correlation between the non zero mutual information score variables.
4. In highly corrrelated variable > 0.7 take the variable with higher mutual information score.

#### Categorical variable:

1. Perform oneway ANOVA test on categorical variables and target.
2. Take variables with p value < 0.05.

#### The flow chart below shows the steps taken to select features.

![](flowchart/features_selection.jpeg)

## Model building

### The following steps were taken to build the model:

1. Build a baseline model on the selected features. Multiple regression model was used as the baseline model.
2. Perform hyperparameter tuning on random forest model using random search cv.
3. Build Random forest model on best parameters from random search cv.
4. Build XGBoost model.

#### The flow chart below shows the steps taken to build the model.

![](flowchart/model%20training.jpeg)

## Model evaluation

### The models were evaluted on test data using the following metrics:

1. RMSE, VIF, mean_residuals
2. OOB error, RMSE for random forest model.
3. RMSE for XGBoost model.

#### The flow chart shows the evaluation metric from the model.

![](flowchart/model%20evaluation.jpeg)

### Folder structure

#### Data

1. interim - Contains the preprocessed data, data from feature engineering and feature selection.
2. processed - Contains the final data used for model building.
3. raw - Contains the raw data and merged data

#### Flowchart

1. Contains the flowchart for each step of the project.

#### logs

1. Contains the logs for each step of the project.

#### models

1. eval: Contains the evaluation metrics for each model.
2. plots: Contains the plots for each model.

#### reports

1. Contains plots and tables for the project.

#### src

1. data - Contains the code for data preparation
2. features - Contains the code for preprocessing, feature engineering, feature selection
3. models - Contains the code for model building and evaluation
4. utils - Contains the code for helper functions

#### config.yml - Contains the configuration for the project

#### main.py - Contains the code to run the project

#### requirements.txt - Contains the required packages for the project

## How to run the code

1. Clone the repository
2. Create a virtual environment
3. Install the requirements
4. Run main.py
