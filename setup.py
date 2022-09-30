from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    install_requires=['pandas == 1.4.4',
                      'logzero == 1.7.0',
                      'matplotlib == 3.5.3',
                      'mlxtend == 0.19.0',
                      'numpy == 1.19.5',
                      'pandas == 1.4.4',
                      'pycaret == 2.2.2',
                      'PyYAML == 6.0',
                      'scikit_learn == 1.1.2',
                      'scipy == 1.5.4',
                      'seaborn == 0.11.2',
                      'xgboost == 1.6.2',
                      'protobuf~=3.19.0',
                      'shap == 0.41.0',
                      'joblib == 1.1.0',
                      'uvicorn[standard]',
                      'azureml-mlflow',
                      'mlflow',
                      'ipython-genutils',
                      'ipykernel == 5.5.5',
                      'papermill == 2.3.3',
                      'uvicorn == 0.14.0',
                      'fastapi == 0.68.0',
                      ]
    version='0.1.1',
    description='Coding challenge',
    author='Viraj Pawar',


)
