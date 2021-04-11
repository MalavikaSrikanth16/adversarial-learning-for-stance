#!/usr/bin/env bash


train_data=data/twitter_testDT_seenval/development_setup_unlab/train.csv
dev_data=data/twitter_testDT_seenval/development_setup_unlab/validation.csv

if [ $1 == 1 ]
then
    echo "training model with early stopping and $3 warm-up epochs and NO cross-entropy weighting"
    python train_model_gens.py -s $2 -i ${train_data} -d ${dev_data} -p $3 -t twitter-topic-TRN-semi-sup.vocab.pkl -k $4

else
    echo "Doing nothing"
fi
