#!/bin/bash
cd ~/Dropbox/Purdue/CS578/final_project/work/senti/
export PYTHONPATH=${PYTHONPATH}:'pwd'


python sdfpreprocess.py ../data/dataset_500_train_small.data train_small.data

python sdfpreprocess.py ../data/dataset_500_val_small.data val_small.data

python sdfpreprocess.py ../data/dataset_500_test_small.data test_small.data
