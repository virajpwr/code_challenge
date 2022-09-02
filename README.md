# The src folder contains preprocessing, feature selection, feature engineering, data preparation, training script.

# The data folder contains the data used for training and testing.

# The models folder contains the trained models.

## Process:

The process of training the model is as follows:

1. Preprocessing: The missing values were imputed using the median value if continous or with median if discrete value.
2. feature engineering: Gauss rank transformation was used to transform the target variable to normal distribution.
3. feature selection: Feature selection was done based on corelation and mutual information score and used recursively do feature selection using XGBoost.
4. training: The model was trained using XGBRegressor. Hyperparameter tuning was done using RandomizedSearchCV and learning rate was set to 0.02.
5. Root mean squared error was used as the evaluation metric. The model was trained on 3 folds and the average of the 3 folds was used as the final score. The RMSE score is 0.8

Learning rate was tuned using grid search and the best learning rate was used for training the model.

### Steps to run the code

1. Clone the repository
2. Create a virtual environment
3. activate the virtual environment
4. Install the requirements: pip install -r requirements.txt
5. Run the main.py file: python main.py or run 'sh run.sh' to run the code.
