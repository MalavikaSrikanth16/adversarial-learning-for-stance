#!/usr/bin/env bash

train_data=data/twitter_testDT_seenval/development_setup/train.csv
dev_data=data/twitter_testDT_seenval/development_setup/validation.csv

echo "training bicond model on twitter data"
python -u train_model.py -s $1 -i ${train_data} -d ${dev_data} -k $2
