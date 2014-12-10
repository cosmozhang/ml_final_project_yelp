#!/bin/bash
cd ~/Dropbox/Purdue/CS578/final_project/work/senti/
export PYTHONPATH=${PYTHONPATH}:'pwd'


python sdfpreprocess.py ../data/dataset_500_train.data train.data

python sdfpreprocess.py ../data/dataset_500_val.data val.data

python sdfpreprocess.py ../data/dataset_500_test.data test.data
