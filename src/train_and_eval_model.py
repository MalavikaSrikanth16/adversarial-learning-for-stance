import torch, sys, os, argparse, time
sys.path.append('./modeling')
import baseline_models as bm
import data_utils, model_utils, datasets
import loss_fn as lf
import torch.optim as optim
import torch.nn as nn
from itertools import chain
import pandas as pd
import copy
from transformers import get_linear_schedule_with_warmup

SEED = 0
LOCAL = True
use_cuda = torch.cuda.is_available()


def train(model_handler, num_epochs, verbose=True, dev_data=None, num_warm=0, phases=False, is_adv=True):
    '''
    Trains the given model using the given data for the specified
    number of epochs. Prints training loss and evaluation starting
    after 10 epochs. Saves at most 10 checkpoints plus a final one.
    :param model_handler: a holder with a model and data to be trained.
                            Assuming the model is a pytorch model.
    :param num_epochs: the number of epochs to train the model for.
    :param verbose: whether or not to print train results while training.
                    Default (True): do print intermediate results.
    '''
    trn_scores_dict = {}
    dev_scores_dict = {}
    for epoch in range(num_epochs):
        if is_adv:
            learning_rate = model_handler.get_learning_rate()
        if phases:
            model_handler.train_step_phases()
        else:
            model_handler.train_step()

        if epoch >= num_warm:
            if verbose:
                # print training loss and training (& dev) scores, ignores the first few epochs
                print("training loss: {}".format(model_handler.loss))
                # eval model on training data
                trn_scores = eval_helper(model_handler, data_name='TRAIN')
                trn_scores_dict[epoch] = copy.deepcopy(trn_scores)
                if is_adv:
                    trn_scores_dict[epoch].update({'lr': copy.deepcopy(learning_rate),
                                                   'rho': copy.deepcopy(model_handler.loss_function.adv_param)})
                # update best scores
                if dev_data is not None:
                    dev_scores = eval_helper(model_handler, data_name='DEV',
                                             data=dev_data)
                    dev_scores_dict[epoch] = copy.deepcopy(dev_scores)
                    model_handler.save_best(scores=dev_scores)
                else:
                    model_handler.save_best(scores=trn_scores)

    print("TRAINED for {} epochs".format(epoch))

    # save final checkpoint
    model_handler.save(num="FINAL")

    # print final training (& dev) scores
    eval_helper(model_handler, data_name='TRAIN')
    if dev_data is not None:
        eval_helper(model_handler,  data_name='DEV', data=dev_data)
    # Can uncomment to save epoch_level_scores
    #save_epoch_level_results_to_csv(trn_scores_dict, dev_scores_dict, model_handler.result_path, model_handler.name, is_adv)


def save_epoch_level_results_to_csv(trn_scores_dict, dev_scores_dict, output_path, name, is_adv):
    '''
    Saves the results from the current epoch to a CSV file
    :param trn_scores_dict: a dictionary containing training scores
    :param dev_scores_dict: a dictionary containing dev set scores
    :param output_path: the path for where to save the scores
    :param name: the prefix for the file name
    :param is_adv: whether or not the scores are from the adversarial  model
    '''
    dev_fscore_overall_list = []
    dev_fscore_seen_list = []
    dev_fscore_unseen_list = []
    train_fscore_overall_list = []
    train_fscore_seen_list = []
    train_fscore_unseen_list = []
    topic_fscore_list = []
    learning_rate_list = []
    rho_list = []
    epochs = []
    for key in trn_scores_dict.keys():
        epochs.append(key)
        if is_adv:
            learning_rate_list.append(trn_scores_dict[key]['lr'])
            rho_list.append(trn_scores_dict[key]['rho'])
            topic_fscore_list.append(trn_scores_dict[key]['topic-f_macro'])
        dev_fscore_overall_list.append(dev_scores_dict[key]['f_macro'])
        dev_fscore_seen_list.append(dev_scores_dict[key]['f-1_macro'])
        dev_fscore_unseen_list.append(dev_scores_dict[key]['f-0_macro'])
        train_fscore_overall_list.append(trn_scores_dict[key]['f_macro'])
        train_fscore_seen_list.append(trn_scores_dict[key]['f-1_macro'])
        train_fscore_unseen_list.append(trn_scores_dict[key]['f-0_macro'])
        
    if is_adv:
        df = pd.DataFrame(list(zip(epochs, learning_rate_list, rho_list, dev_fscore_overall_list, dev_fscore_seen_list,
                                   dev_fscore_unseen_list, topic_fscore_list, train_fscore_overall_list,
                                   train_fscore_seen_list,train_fscore_unseen_list)),
                      columns=['Epoch', 'Learning Rate', 'Rho', 'Dev Fscore overall', 'Dev Fscore seen',
                               'Dev Fscore unseen', 'Topic Fscore', 'Train Fscore overall', 'Train Fscore seen',
                               'Train Fscore unseen'])
    else:
        df = pd.DataFrame(list(zip(epochs, dev_fscore_overall_list, dev_fscore_seen_list, dev_fscore_unseen_list,
                               train_fscore_overall_list, train_fscore_seen_list, train_fscore_unseen_list)),
                      columns=['Epoch', 'Dev Fscore overall', 'Dev Fscore seen', 'Dev Fscore unseen',
                               'Train Fscore overall', 'Train Fscore seen', 'Train Fscore unseen'])
    df.to_csv("{}{}_epoch_level_scores.csv".format(output_path, name), index=False)


def eval_helper(model_handler, data_name, data=None):
    '''
    Helper function for evaluating the model during training.
    Can evaluate on all the data or just a subset of corpora.
    :param model_handler: the holder for the model
    :return: the scores from running on all the data
    '''
    # eval on full corpus
    scores = model_handler.eval_and_print(data=data, data_name=data_name, class_wise=True)
    return scores


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', dest='mode', help='What to do', required=True)
    parser.add_argument('--config_file', dest='config_file', help='Name of the cofig data file', required=False)
    parser.add_argument('--trn_data', dest='trn_data', help='Name of the training data file', required=False)
    parser.add_argument('--dev_data', dest='dev_data', help='Name of the dev data file', default=None, required=False)
    parser.add_argument('--name', dest='name', help='something to add to the saved model name',
                        required=False, default='')
    parser.add_argument('-p', '--num_warm', help='Number of warm-up epochs', required=False,
                        type=int, default=0)
    parser.add_argument('--topics_vocab', dest='topics_vocab', help='Name of the topic file', required=False,
                        type=str, default='twitter-topic.vocab.pkl')
    parser.add_argument('--score_key', dest='score_key', help='Score key for optimization', required=False,
                        default='f_macro')
    parser.add_argument('--saved_model_file_name', dest='saved_model_file_name', required=False, default=None)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

    ####################
    # load config file #
    ####################
    with open(args.config_file, 'r') as f:
        config = dict()
        for l in f.readlines():
            config[l.strip().split(":")[0]] = l.strip().split(":")[1]

    ################
    # load vectors #
    ################
    if not LOCAL:
        vec_path = 'resources'
    else:
        vec_path = 'data/resources'     #Need to set path to vectors here

    if 'bert' not in config['name']:

        vec_name = config['vec_name']
        vec_dim = int(config['vec_dim'])

        vecs = data_utils.load_vectors('{}/{}.vectorsF.npy'.format(vec_path, vec_name),
                                       dim=vec_dim, seed=SEED)

    #############
    # LOAD DATA #
    #############
    # load training data
    vocab_name = '{}/{}.vocabF.pkl'.format(vec_path, vec_name)
    data = datasets.StanceData(args.trn_data, vocab_name, topic_name='{}/{}'.format(vec_path, args.topics_vocab),
                           pad_val=len(vecs) - 1,
                           max_tok_len=int(config.get('max_tok_len', '200')),
                           max_sen_len=int(config.get('max_sen_len', '10')),
                           max_top_len=int(config.get('max_top_len', '5')))
    
    dataloader = data_utils.DataSampler(data,  batch_size=int(config['b']))

    # load dev data if specified
    if args.dev_data is not None:
        dev_data = datasets.StanceData(args.dev_data, vocab_name, topic_name=None,
                                       pad_val=len(vecs) - 1,
                                       max_tok_len=int(config.get('max_tok_len', '200')),
                                       max_sen_len=int(config.get('max_sen_len', '10')),
                                       max_top_len=int(config.get('max_top_len', '5')),
                                       use_tar_in_twe=('use_tar_in_twe' in config))

        dev_dataloader = data_utils.DataSampler(dev_data, batch_size=int(config['b']), shuffle=False)

    else:
        dev_dataloader = None

    # set the optimizer
    if 'optimizer' not in config:
        optim_fn = optim.Adam
    else:
        if config['optimizer'] == 'adamw':
            optim_fn = optim.AdamW
        elif config['optimizer'] == 'sgd':
            optim_fn = optim.SGD
        else:
            print("ERROR with optimizer")
            sys.exit(1)

    lr = float(config.get('lr', '0.001'))
    nl = 3
    adv = False

    # RUN
    print("Using cuda?: {}".format(use_cuda))

    if 'bert' in config['name']:
        batch_args = {'keep_sen': False}
        setup_fn = data_utils.setup_helper_bert_ffnn
        loss_fn = nn.CrossEntropyLoss()

        input_layer = None
        model = bm.JointSeqBERTLayer(nl, use_cuda=use_cuda)

        optimizer = optim.AdamW(model.parameters(), lr=lr)

        num_training_steps = len(data) * int(config['epochs'])
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0.1 * num_training_steps,
                                                    num_training_steps=num_training_steps)

        kwargs = {'model': model, 'embed_model': input_layer, 'dataloader': dataloader,
                  'batching_fn': data_utils.prepare_batch,
                  'batching_kwargs': batch_args, 'name': config['name'],
                  'loss_function': loss_fn,
                  'optimizer': optimizer,
                  'scheduler': scheduler,
                  'setup_fn': setup_fn,
                  'fine_tune': (config.get('fine-tune', 'no') == 'yes')}

        model_handler = model_utils.TorchModelHandler(use_cuda=use_cuda,
                                                      checkpoint_path=config.get('ckp_path', 'data/checkpoints/'),
                                                      result_path=config.get('res_path', 'data/gen-stance/'),
                                                      use_score=args.score_key, save_ckp=args.save_ckp,
                                                      **kwargs)

    elif 'BiCond' in config['name']:
        batch_args = {}
        input_layer = bm.WordEmbedLayer(vecs=vecs, use_cuda=use_cuda)

        setup_fn = data_utils.setup_helper_bicond

        loss_fn = nn.CrossEntropyLoss()

        model = bm.BiCondLSTMModel(int(config['h']), embed_dim=input_layer.dim,
                                   input_dim=(int(config['in_dim']) if 'in_dim' in config['name'] else input_layer.dim),
                                   drop_prob=float(config['dropout']), use_cuda=use_cuda,
                                   num_labels=nl)
        o = optim_fn(model.parameters(), lr=lr)

        bf = data_utils.prepare_batch

        kwargs = {'model': model, 'embed_model': input_layer, 'dataloader': dataloader,
                  'batching_fn': bf,
                  'batching_kwargs': batch_args, 'name': config['name'] + args.name,
                  'loss_function': loss_fn,
                  'optimizer': o,
                  'setup_fn': setup_fn,
                  'blackout_start': int(config['blackout_start']),
                  'blackout_stop': int(config['blackout_stop'])}

        model_handler = model_utils.TorchModelHandler(use_cuda=use_cuda,
                                                      checkpoint_path=config.get('ckp_path', 'data/checkpoints/'),
                                                      result_path=config.get('res_path','data/gen-stance/'),
                                                      **kwargs)

    elif 'BasicAdv' in config['name']:
        batch_args = {}
        input_layer = bm.WordEmbedLayer(vecs=vecs, use_cuda=use_cuda)
        setup_fn = data_utils.setup_helper_adv

        loss_fn = lf.AdvBasicLoss(trans_dim=2*int(config['h']), trans_param=float(config['trans_w']),
                                  num_no_adv=float(config['num_na']),
                                  tot_epochs=int(config['epochs']),
                                  rho_adv=('rho_adv' in config),
                                  gamma=float(config.get('gamma', 10.0)),
                                  semi_sup=('semi_sup' in config),
                                  use_cuda=use_cuda)

        enc_params = {'h': int(config['h']), 'embed_dim': input_layer.dim, 'drop_prob' : float(config['dropout'])}

        model = bm.AdversarialBasic(enc_params=enc_params, enc_type=config['enc'],
                                    stance_dim=int(config['sd']), topic_dim=int(config['td']),
                                    num_labels=nl, num_topics=int(config['num_top']),
                                    drop_prob=float(config['dropout']),
                                    use_cuda=use_cuda)
        
        if 'optimizer' not in config:
            #Adam optimizer
            o_main = optim_fn(chain(model.enc.parameters(),
                                model.recon_layer.parameters(),
                                model.topic_recon_layer.parameters(),
                                model.trans_layer.parameters(),
                                model.stance_classifier.parameters()),
                          lr=lr,
                          weight_decay=float(config.get('l2_main', '0')))
            o_adv = optim_fn(model.topic_classifier.parameters(),
                             lr=lr,
                             weight_decay=float(config.get('l2_adv', '0')))
        elif config['optimizer'] == 'sgd':
            #SGD optimizer
            o_main = optim_fn(chain(model.enc.parameters(),
                                    model.recon_layer.parameters(),
                                    model.topic_recon_layer.parameters(),
                                    model.trans_layer.parameters(),
                                    model.stance_classifier.parameters()),
                              lr=lr,
                              weight_decay=float(config.get('l2_main', '0')),
                              momentum=0.9,
                              nesterov=True)
            o_adv = optim_fn(model.topic_classifier.parameters(),
                             lr=lr,
                             weight_decay=float(config.get('l2_adv', '0')),
                             momentum=0.9,
                             nesterov=True)

        kwargs = {'model': model, 'embed_model': input_layer, 'dataloader': dataloader,
                  'batching_fn': data_utils.prepare_batch_adv,
                  'batching_kwargs': batch_args, 'name': config['name'] + '-{}'.format(config['enc']) + args.name,
                  'loss_function': loss_fn,
                  'optimizer': o_main,
                  'adv_optimizer': o_adv,
                  'setup_fn': setup_fn,
                  'tot_epochs': int(config['epochs']),
                  'initial_lr': lr,
                  'alpha': float(config.get('alpha', 10.0)),
                  'beta': float(config.get('beta', 0.75)),
                  'num_constant_lr': float(config['num_constant_lr']),
                  'batch_size': int(config['b']),
                  'blackout_start': int(config['blackout_start']),
                  'blackout_stop': int(config['blackout_stop'])}

        model_handler = model_utils.AdvTorchModelHandler(use_score=args.score_key, use_cuda=use_cuda,
                                                         checkpoint_path=config.get('ckp_path', 'data/checkpoints/'),
                                                         result_path=config.get('res_path', 'data/gen-stance/'),
                                                         opt_for=config.get('opt', 'score_key'),
                                                         **kwargs)

    if args.mode == 'train':
        # Train model
        start_time = time.time()
        train(model_handler, int(config['epochs']), dev_data=dev_dataloader,
             num_warm=args.num_warm, phases=('phases' in config),is_adv=adv)
        print("[{}] total runtime: {:.2f} minutes".format(config['name'], (time.time() - start_time)/60.))

    elif args.mode == 'eval':
        # Evaluate saved model
        model_handler.load(filename=args.saved_model_file_name)
        eval_helper(model_handler,'DEV',data=dev_dataloader)