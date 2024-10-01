# survInterpretability
_survInterpretability_ is a framework for interpreting survival models

## Quick description
_survInterpretability_ is a Python 3 package 

## Use cases

## Installation
Clone the repository, then inside the folder, use a `virtualenv` to install the requirements
```shell script
git clone https://github.com/lucasducrot/survInterpretability.git
cd survInterpretability

# Install packages with Pip
# If your default interpreter is Python3:
virtualenv .venv_surv_interpret
# If your default interpreter is Python2, you can explicitly target Python3 with:
virtualenv -p python3.12 .venv_surv_interpret

source .venv_surv_interpret/bin/activate
```
Then, to download all required modules and initialize the project run the following commands:
```shell script
pip3 install -r requirements.txt
pip3 install -e .
```
The second command installs the project as a package, making the main module importable from anywhere.

## Install packages with conda environment
```shell script
conda env create -n .venv_surv_interpret python=3.12 -f requirements_conda.yml

conda activate .venv_surv_interpret
```

Then, to add the virtual environment to jupyter kernel:
```shell script
python -m ipykernel install --user --name=".venv_surv_interpret"
```


## Notebook tutorial

The Jupyter notebook "tutorials" gives useful example of how to use the framework