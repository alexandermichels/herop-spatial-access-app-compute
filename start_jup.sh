#!/bin/bash
source ~/.bashrc
#conda activate /data/keeling/a/michels9/mambaforge/envs/mlaccess
conda activate tcm
which python3
jupyter notebook --port=${1} --no-browser --ip=127.0.0.1
