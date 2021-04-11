# Adversarial Learning for Zero-Shot Stance Detection on Social Media

## Requirements
python 3.7.6 <br/>
transformers 3.4.0 <br/>
pytorch 1.5.1 <br/>
numpy 1.18.1 <br/>
pandas 1.0.3 <br/>
scipy 1.4.1


## Training TOAD
cd src/ 

Create folder data and within it create a folder named resources. 

In data/resources, place pretrained GloVe word embeddings and topic dictionary (which maps topics in training data to indices).

Run 
```angular2html
python train_and_eval_model.py --mode "train" --config_file <config_name> --trn_data <train_data> --dev_data <dev_data> --score_key <score_key> --topics_vocab <topic_dictionary> --mode train 
```
For example:
```angular2html
python train_and_eval_model.py --mode "train" --config_file data/config-0.txt --trn_data data/twitter_testDT_seenval/development_setup/train.csv --dev_data data/twitter_testDT_seenval/development_setup/validation.csv --score_key f_macro --topics_vocab twitter-topic-TRN-semi-sup.vocab.pkl --mode train 
```
Score key is evaluated on the development data and used for saving the best model across epochs.

Config file for TOAD should follow the format of our example TOAD config file - src/config_example_toad.txt

## Evaluating a saved TOAD model
To evaluate a saved model on test_data, run 
```angular2html
python train_and_eval_model.py --mode "eval" --config_file <config_name> --trn_data <train_data> --dev_data <test_data> --topics_vocab <topic_dictionary> --saved_model_file_name <saved_model_file_name> --mode eval 
```

For example:
```angular2html
python train_and_eval_model.py --mode "eval" --config_file data/config-0.txt --trn_data data/twitter_testDT_seenval/development_setup/train.csv --dev_data data/twitter_testDT_seenval/test_setup/test.csv --saved_model_file_name data/checkpoints/DT_checkpoint.tar --topics_vocab twitter-topic-TRN-semi-sup.vocab.pkl --mode eval 
```

## Baseline models
### BiCond
Run
```angular2html
python train_and_eval_model.py --mode "train" --config_file <config_name> --trn_data <train_data> --dev_data <dev_data> --score_key <score_key>
```

Config file should follow the format of our example BiCond config file - src/config_example_bicond.txt
### BERT
Run
```angular2html
python train_and_eval_model.py --mode "train" --config_file <config_name> --trn_data <train_data> --dev_data <dev_data> --score_key <score_key>
```

## Hyperparameter search for TOAD
Run
```angular2html
python hyperparam_selection.py -m 1 -s <config_file_for_hyperparam_tuning> -k <score_key>
```

Config file should follow the format of our example TOAD hyperparameter search config file -  src/hyperparam-twitter-adv.txt



