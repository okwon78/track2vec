#!/bin/bash

sudo apt-get -y update
sudo apt-get -y install git clang g++ cmake awscli zlib1g python3-pip

#sudo yum update -y
#sudo yum groupinstall -y 'Development Tools'

git config --global core.editor "vim"

PYTHON_VERSION=3.7.3
PYENV_GIT_URL=https://github.com/pyenv/pyenv.git
PYENV_LOCAL_PATH=${HOME}/.pyenv

git clone ${PYENV_GIT_URL} ${PYENV_LOCAL_PATH}

echo 'export PYENV_ROOT="'${PYENV_LOCAL_PATH}'"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc

. ~/.bashrc

pyenv install ${PYTHON_VERSION}
pyenv global ${PYTHON_VERSION}


pip install annoy awscli