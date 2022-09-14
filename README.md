#### The src folder contains preprocessing, feature selection, feature engineering, data preparation, training script.

##### The data folder contains the data used for training and testing.

##### The models folder contains the trained models.

## Process:

The process of training the model is as follows:

1. Preprocessing: The missing values were imputed using the median value if continous or with median if discrete value.
2. feature engineering: Gauss rank transformation was used to transform the target variable to normal distribution.
3. feature selection: Feature selection was done based on corelation and mutual information score and used recursively do feature selection using XGBoost.
4. training: The model was trained using XGboost. Trained XGboost baseline model with max_depth = 18, learning rate = 0.02
   Hyperparameter tuning using RandomizedSearchCV is available in the code and can be utilized to get best hyperparameters for further tuning of the model.
5. Root mean squared error was used as the evaluation metric. The RMSE score is 0.8
6. Check RMSE score in eval folder
7. The SHAP bar plot showing important features is available in plots folder and shown below.
   ![Figure_1](https://user-images.githubusercontent.com/36328852/188136359-b2faacfe-feb7-43ae-b9b4-b2ae128a555a.png)
   ![Figure_1](https://user-images.githubusercontent.com/36328852/190073735-aae60fd5-de70-4f56-a4f7-6054c424eaf1.png)



### Steps to run the code

1. Clone the repository
2. Create a virtual environment
3. activate the virtual environment
4. Install the requirements: pip install -r requirements.txt
5. Run the main.py file: python main.py or run 'sh run.sh' to run the code.
