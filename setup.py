from importlib.metadata import entry_points
from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="code-challenge",
    version="0.1.0",
    author="Viraj Pawar",
    author_email="virajp.mail@gmail.com",
    description=(
        "Code to train model on training data and predict on test data"
    ),
    install_requires=requirements,
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "train=src.__main__:main",
        ]
    }
)