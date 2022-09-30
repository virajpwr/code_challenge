from azure.ai.ml import MLClient
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component
import os
from pathlib import Path
from mldesigner import command_component, Input, Output

ml_client = MLClient.from_config(credential = './.azureml/credentials.json')
print(ml_client)