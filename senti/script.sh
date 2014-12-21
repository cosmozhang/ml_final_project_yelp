#!/bin/bash
cd ~/Dropbox/Purdue/CS578/final_project/work/senti/
export PYTHONPATH=${PYTHONPATH}:'pwd'


python sdfpreprocess.py ../data/dataset_200_resample_train_small.data 200r_small_train.data

python sdfpreprocess.py ../data/dataset_200_resample_val_small.data 200r_small_val.data

python sdfpreprocess.py ../data/dataset_200_resample_test_small.data 200r_small_test.data
