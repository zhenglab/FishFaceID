#!/bin/bash
ANACONDA_HOME=/mnt/8TDisk1/zhenglab/sunnaizhe/envs/miniconda310-vmamba
# ANACONDA_HOME=/mnt/8TDisk1/zhenglab/sunnaizhe/envs/miniconda310-vim
# ANACONDA_HOME=/mnt/8TDisk1/zhenglab/sunnaizhe/envs/anaconda38-openmm
ANACONDA_PYPATH=${ANACONDA_HOME}/lib/python3.10/site-packages

export PATH=${ANACONDA_HOME}/bin:$PATH
export PYTHONPATH=${ANACONDA_PYPATH}:$PYTHONPATH


