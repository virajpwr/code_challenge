
from azureml.core import Workspace, Dataset

subscription_id = '3e42da7d-fb30-45ac-96b8-f97ea1531dbd'
resource_group = 'virajpawar'
workspace_name = 'virajpawar'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='merged_data')
dataset = dataset.to_pandas_dataframe()
print(dataset.columns)
