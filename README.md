# Tensorflow Keras Prototype


## Summary

* sample project of Tensorflow Keras with pyenv pipenv
* mnist MLP (Multilayer Perceptron) model as a baseline
* [TODO] mnist CNN model


## Setup Python environment

### install pyenv pipenv

```bash
# if you haven't set these
export LC_ALL='en_US.UTF-8'
export LANG='en_US.UTF-8'
# setup python env tools
brew install pyenv pipenv
echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
echo 'eval "$(pipenv --completion)"' >> ~/.bash_profile
echo 'export PIPENV_VENV_IN_PROJECT=true' >> ~/.bash_profile
source ~/.bash_profile
```

### install packages

```bash
pipenv install
# check installed packages
pipenv graph
# run shell and python
pipenv shell
python
# or run python directly
pipenv run python
```


## Run jupyter notebook

use `#%%` to run in VSCode .py files

```bash
export PYTHONPATH=${PWD}/src
code .
# open src/baseline/baseline.py
```

or run jupyter lab server

```bash
pipenv run jupyter lab --ip=0.0.0.0 --no-browser
open localhost:8888/lab
```


## Run Tensorboard

```bash
tensorboard --logdir=/tmp/tf_log/
open http://localhost:6006/
```


## (reference) pipenv misc commands

```bash
# install using requirements.txt
pipenv install -r requirements.txt
# requirements.txt style output
pipenv run pip freeze
# update packages
pipenv update --outdated
pipenv update
# downgrade (for example tornado)
pipenv install "tornado<6"
# initialize .venv (remove all packages)
pipenv --python 3.6
# remove .venv
pipenv --rm
```


## Reference

* Keras reference book
  - https://github.com/oreilly-japan/deep-learning-with-keras-ja
* VSCode Jupyter Support
  - https://devblogs.microsoft.com/python/python-in-visual-studio-code-october-2018-release/
  - http://miso-soup3.hateblo.jp/entry/2018/12/20/223705
