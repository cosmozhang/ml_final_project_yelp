#!/bin/bash
cd ~/Dropbox/Purdue/CS578/final_project/work/senti/
export PYTHONPATH=${PYTHONPATH}:'pwd'

for (( i=1; i<11; i++ ))
do
    echo $i
    python sdfpreprocess.py ../data/dataset_500.data ${i}
done
