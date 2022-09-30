# Upgrade pip
py -m pip install --upgrade pip

# Install virtualenv
py -m pip install virtualenv

# Create virtual environment
py -m virtualenv venv

# Activate virtual environment
. venv/Scripts/activate

# Install requirements
py -m pip install -r requirements.txt

# run training and evaluate model and print RMSE score
py main.py