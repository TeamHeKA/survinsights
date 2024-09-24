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

# If your default interpreter is Python3:
virtualenv .venv_surv_interpret
# If your default interpreter is Python2, you can explicitly target Python3 with:
virtualenv -p python3.7 .venv_surv_interpret

source .venv_surv_interpret/bin/activate
```
Then, to download all required modules and initialize the project run the following commands:
```shell script
pip install -r requirements.txt
pip install -e .
```
The second command installs the project as a package, making the main module importable from anywhere.
Then, to add the virtual eviroment to jupyter kernel:
```shell script
python -m ipykernel install --user --name=".venv_surv_interpret"
```


## Notebook tutorial

The Jupyter notebook "tutorials" gives useful example of how to use the framework