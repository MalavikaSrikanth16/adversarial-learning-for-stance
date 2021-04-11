import torch, pickle, time, json, copy
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

class TorchModelHandler:
    '''
    Class that holds a model and provides the functionality to train it,
    save it, load it, and evaluate it. The model used here is assumed to be
    written in pytorch.
    '''
    # def __init__(self, model, loss_function, dataloader, optimizer, name, num_ckps=10,
    #              use_score='f_macro', device='cpu', use_last_batch=True):
    def __init__(self, num_ckps=10, use_score='f_macro', use_cuda=False,
                 checkpoint_path='data/checkpoints/',
                 result_path='data/', **params):
        super(TorchModelHandler, self).__init__()
        # data fields
        self.model = params['model']
        self.embed_model = params['embed_model']
        self.dataloader = params['dataloader']
        self.batching_fn = params['batching_fn']
        self.batching_kwargs = params['batching_kwargs']
        self.setup_fn = params['setup_fn']

        self.num_labels = self.model.num_labels
        self.name = params['name']

        # optimization fields
        self.loss_function = params['loss_function']
        self.optimizer = params['optimizer']
        self.fine_tune = params.get('fine_tune', False)

        # stats fields
        self.checkpoint_path = checkpoint_path
        self.checkpoint_num = 0
        self.num_ckps = num_ckps
        self.epoch = 0

        self.result_path = result_path

        # evaluation fields
        self.score_dict = dict()
        self.max_score = 0.
        self.max_lst = []  # to keep top 5 scores
        self.score_key = use_score
        self.blackout_start = params['blackout_start']
        self.blackout_stop = params['blackout_stop']

        # GPU support
        self.use_cuda = use_cuda
        if self.use_cuda:
            # move model and loss function to GPU, NOT the embedder
            self.model = self.model.to('cuda')
            self.loss_function = self.loss_function.to('cuda')

    def save_best(self, data=None, scores=None, data_name=None, class_wise=False):
        '''
        Evaluates the model on data and then updates the best scores and saves the best model.
        :param data: data to evaluate and update based on. Default (None) will evaluate on the internally
                        saved data. Otherwise, should be a DataSampler. Only used if scores is not None.
        :param scores: a dictionary of precomputed scores. Default (None) will compute a list of scores
                        using the given data, name and class_wise flag.
        :param data_name: the name of the data evaluating and updating on. Only used if scores is not None.
        :param class_wise: lag to determine whether to compute class-wise scores in
                            addition to macro-averaged scores. Only used if scores is not None.
        '''
        if scores is None:
            # evaluate and print
            scores = self.eval_and_print(data=data, data_name=data_name,
                                         class_wise=class_wise)
        scores = copy.deepcopy(scores)  # copy the scores, otherwise storing a pointer which won't track properly

        if self.epoch in range(self.blackout_start, self.blackout_stop):
            return
            # update list of top scores
        curr_score = scores[self.score_key]
        score_updated = False
        if len(self.max_lst) < 5:
            score_updated = True
            if len(self.max_lst) > 0:
                prev_max = self.max_lst[-1][0][self.score_key] # last thing in the list
            else:
                prev_max = curr_score
            self.max_lst.append((scores, self.epoch - 1))
        elif curr_score > self.max_lst[0][0][self.score_key]: # if bigger than the smallest score
            score_updated = True
            prev_max = self.max_lst[-1][0][self.score_key] # last thing in the list
            self.max_lst[0] = (scores, self.epoch - 1) #  replace smallest score

        # update best saved model and file with top scores
        if score_updated:
            # prev_max = self.max_lst[-1][0][self.score_key]
            # sort the scores
            self.max_lst = sorted(self.max_lst, key=lambda p: p[0][self.score_key])  # lowest first
            # write top 5 scores
            f = open('{}{}.top5_{}.txt'.format(self.result_path, self.name, self.score_key), 'w')  # overrides
            for p in self.max_lst:
                f.write('Epoch: {}\nScore: {}\nAll Scores: {}\n'.format(p[1], p[0][self.score_key],
                                                                            json.dumps(p[0])))
            # save best model step, if its this one
            print(curr_score, prev_max)
            if curr_score > prev_max or self.epoch == 1:
                self.save(num='BEST')

    def save(self, num=None):
        '''
        Saves the pytorch model in a checkpoint file.
        :param num: The number to associate with the checkpoint. By default uses
                    the internally tracked checkpoint number but this can be changed.
        '''
        if num is None:
            check_num = self.checkpoint_num
        else: check_num = num

        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss
        }, '{}ckp-{}-{}.tar'.format(self.checkpoint_path, self.name, check_num))

        if num is None:
            self.checkpoint_num = (self.checkpoint_num + 1) % self.num_ckps

    def load(self, filename='data/checkpoints/ckp-[NAME]-FINAL.tar', use_cpu=False):
        '''
        Loads a saved pytorch model from a checkpoint file.
        :param filename: the name of the file to load from. By default uses
                        the final checkpoint for the model of this' name.
        '''
        filename = filename.replace('[NAME]', self.name)
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def train_step(self):
        '''
        Runs one epoch of training on this model.
        '''
        print("[{}] epoch {}".format(self.name, self.epoch))
        self.model.train()
        self.loss = 0.  # clear the loss
        start_time = time.time()
        for i_batch, sample_batched in enumerate(self.dataloader):

            # zero gradients before EVERY optimizer step
            self.model.zero_grad()
            if self.tune_embeds:
                self.embed_model.zero_grad()

            y_pred, labels = self.get_pred_with_grad(sample_batched)

            label_tensor = torch.tensor(labels)
            if self.use_cuda:
                # move labels to cuda if necessary
                label_tensor = label_tensor.to('cuda')

            if self.dataloader.weighting:
                batch_loss = self.loss_function(y_pred, label_tensor)
                weight_lst = torch.tensor([self.dataloader.topic2c2w[b['ori_topic']][b['label']]
                                           for b in sample_batched])
                if self.use_cuda:
                    weight_lst = weight_lst.to('cuda')
                graph_loss = torch.mean(batch_loss * weight_lst)
            else:
                graph_loss = self.loss_function(y_pred, label_tensor)

            # self.loss = graph_loss.item()
            self.loss += graph_loss.item()  # update loss

            graph_loss.backward()

            self.optimizer.step()

        end_time = time.time()
        # self.dataloader.reset()
        print("   took: {:.1f} min".format((end_time - start_time)/60.))
        self.epoch += 1

    def compute_scores(self, score_fn, true_labels, pred_labels, class_wise, name):
        '''
        Computes scores using the given scoring function of the given name. The scores
        are stored in the internal score dictionary.
        :param score_fn: the scoring function to use.
        :param true_labels: the true labels.
        :param pred_labels: the predicted labels.
        :param class_wise: flag to determine whether to compute class-wise scores in
                            addition to macro-averaged scores.
        :param name: the name of this score function, to be used in storing the scores.
        '''
        labels = [i for i in range(2)]
        n = float(len(labels))

        vals = score_fn(true_labels, pred_labels, labels=labels, average=None)
        self.score_dict['{}_macro'.format(name)] = sum(vals) / n

        if class_wise:
            self.score_dict['{}_anti'.format(name)] = vals[0]
            self.score_dict['{}_pro'.format(name)] = vals[1]
            if n > 2:
                self.score_dict['{}_none'.format(name)] = vals[2]

    def eval_model(self, data=None, class_wise=False, data_name=None):
        '''
        Evaluates this model on the given data. Stores computed
        scores in the field "score_dict". Currently computes macro-averaged
        F1 scores, precision and recall. Can also compute scores on a class-wise basis.
        :param data: the data to use for evaluation. By default uses the internally stored data
                    (should be a DataSampler if passed as a parameter).
        :param class_wise: flag to determine whether to compute class-wise scores in
                            addition to macro-averaged scores.
        :return: a map from score names to values
        '''
        pred_labels, true_labels, t2pred, marks = self.predict(data)
        self.score(pred_labels, true_labels, class_wise, t2pred, marks)

        return self.score_dict

    def predict(self, data=None):
        all_y_pred = None
        all_labels = None
        all_marks = None
        all_tar_in_twe = None

        self.model.eval()
        self.loss = 0.

        if data is None:
            data = self.dataloader

        t2pred = dict()
        for sample_batched in data:
            with torch.no_grad():
                # print(sample_batched)
                y_pred, labels = self.get_pred_noupdate(sample_batched)

                label_tensor = torch.tensor(labels)
                if self.use_cuda:
                    # move labels to cuda if necessary
                    label_tensor = label_tensor.to('cuda')  # .cuda()
                self.loss += self.loss_function(y_pred, label_tensor).item()

                y_pred_arr = y_pred.detach().cpu().numpy()
                ls = np.array(labels)

                m = [b['seen'] for b in sample_batched]
                tar_in_twe = [b['target_in_tweet'] for b in sample_batched]

                for bi, b in enumerate(sample_batched):
                    t = b['ori_topic']
                    t2pred[t] = t2pred.get(t, ([], []))
                    t2pred[t][0].append(y_pred_arr[bi, :])
                    t2pred[t][1].append(ls[bi])

                if all_y_pred is None:
                    all_y_pred = y_pred_arr
                    all_labels = ls
                    all_marks = m
                    all_tar_in_twe = tar_in_twe
                else:
                    all_y_pred = np.concatenate((all_y_pred, y_pred_arr), 0)
                    all_labels = np.concatenate((all_labels, ls), 0)
                    all_marks = np.concatenate((all_marks, m), 0)
                    all_tar_in_twe = np.concatenate((all_tar_in_twe, tar_in_twe), 0)

        for t in t2pred:
            t2pred[t] = (np.argmax(t2pred[t][0], axis=1), t2pred[t][1])

        if None not in all_tar_in_twe:
            all_tar_in_twe = np.array(all_tar_in_twe)
            tar_in_twe_mask = np.column_stack((np.zeros(len(all_tar_in_twe)), np.zeros(len(all_tar_in_twe)), all_tar_in_twe))
            all_y_pred = np.where(tar_in_twe_mask == 1, -np.inf, all_y_pred)
        pred_labels = all_y_pred.argmax(axis=1)
        true_labels = all_labels
        return pred_labels, true_labels, t2pred, all_marks

    def eval_and_print(self, data=None, data_name=None, class_wise=False):
        '''
        Evaluates this model on the given data. Stores computed
        scores in the field "score_dict". Currently computes macro-averaged.
        Prints the results to the console.
        F1 scores, precision and recall. Can also compute scores on a class-wise basis.
        :param data: the data to use for evaluation. By default uses the internally stored data
                    (should be a DataSampler if passed as a parameter).
        :param data_name: the name of the data evaluating.
        :param class_wise: flag to determine whether to compute class-wise scores in
                            addition to macro-averaged scores.
        :return: a map from score names to values
        '''
        # Passing data_name to eval_model as evaluation of adv model on train and dev are different
        scores = self.eval_model(data=data, class_wise=class_wise, data_name=data_name)
        print("Evaling on \"{}\" data".format(data_name))
        for s_name, s_val in scores.items():
            print("{}: {}".format(s_name, s_val))
        return scores

    def score(self, pred_labels, true_labels, class_wise, t2pred, marks, topic_wise=False):
        '''
        Helper Function to compute scores. Stores updated scores in
        the field "score_dict".
        :param pred_labels: the predicted labels
        :param true_labels: the correct labels
        :param class_wise: flag to determine whether to compute class-wise scores in
                            addition to macro-averaged scores.
        '''
        # calculate class-wise and macro-averaged F1
        self.compute_scores(f1_score, true_labels, pred_labels, class_wise, 'f')
        # calculate class-wise and macro-average precision
        self.compute_scores(precision_score, true_labels, pred_labels, class_wise, 'p')
        # calculate class-wise and macro-average recall
        self.compute_scores(recall_score, true_labels, pred_labels, class_wise, 'r')

        for v in [1, 0]:
            tl_lst = []
            pl_lst = []
            for m, tl, pl in zip(marks, true_labels, pred_labels):
                if m != v: continue
                tl_lst.append(tl)
                pl_lst.append(pl)
            self.compute_scores(f1_score, tl_lst, pl_lst, class_wise, 'f-{}'.format(v))
            self.compute_scores(precision_score, tl_lst, pl_lst, class_wise, 'p-{}'.format(v))
            self.compute_scores(recall_score, tl_lst, pl_lst, class_wise, 'r-{}'.format(v))

        if topic_wise:
            for t in t2pred:
                self.compute_scores(f1_score, t2pred[t][1], t2pred[t][0], class_wise,
                                    '{}-f'.format(t))

    def get_pred_with_grad(self, sample_batched):
        '''
        Helper function for getting predictions while tracking gradients.
        Used for training the model.
        OVERRIDES: super method.
        :param sample_batched: the batch of data samples
        :return: the predictions for the batch (as a tensor) and the true
                    labels for the batch (as a numpy array)
        '''
        args = self.batching_fn(sample_batched, **self.batching_kwargs)

        if not self.fine_tune:
            # EMBEDDING
            embed_args = self.embed_model(**args)
            args.update(embed_args)

            # PREDICTION
            y_pred = self.model(*self.setup_fn(args, self.use_cuda))

        else:
            y_pred = self.model(**args)

        labels = args['labels']

        return y_pred, labels

    def get_pred_noupdate(self, sample_batched):
        '''
        Helper function for getting predictions without tracking gradients.
        Used for evaluating the model or getting predictions for other reasons.
        OVERRIDES: super method.
        :param sample_batched: the batch of data samples
        :return: the predictions for the batch (as a tensor) and the true labels
                    for the batch (as a numpy array)
        '''
        args = self.batching_fn(sample_batched, **self.batching_kwargs)

        with torch.no_grad():
            if not self.fine_tune:
                # EMBEDDING
                embed_args = self.embed_model(**args)
                args.update(embed_args)

                # PREDICTION
                y_pred = self.model(*self.setup_fn(args, self.use_cuda))
            else:
                y_pred = self.model(**args)

            labels = args['labels']

        return y_pred, labels


class AdvTorchModelHandler(TorchModelHandler):
    def __init__(self, num_ckps=10, use_score='f_macro', use_cuda=False, use_last_batch=True,
                 num_gpus=None, checkpoint_path='data/checkpoints/',
                 result_path='data/', opt_for='score_key', **params):
        TorchModelHandler.__init__(self, num_ckps=num_ckps, use_score=use_score, use_cuda=use_cuda,
                                   use_last_batch=use_last_batch, num_gpus=num_gpus,
                                   checkpoint_path=checkpoint_path, result_path=result_path,
                                   opt_for=opt_for,
                                   **params)
        self.adv_optimizer = params['adv_optimizer']
        self.tot_epochs = params['tot_epochs']
        self.initial_lr = params['initial_lr']
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.num_constant_lr = params['num_constant_lr']
        self.batch_size = params['batch_size']

    def adjust_learning_rate(self, epoch):
        if epoch >= self.num_constant_lr:
            tot_epochs_for_calc = self.tot_epochs - self.num_constant_lr
            epoch_for_calc = epoch - self.num_constant_lr
            p = epoch_for_calc / tot_epochs_for_calc
            new_lr = self.initial_lr / ((1 + self.alpha * p) ** self.beta)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            for param_group in self.adv_optimizer.param_groups:
                param_group['lr'] = new_lr

    def get_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
            break
        return lr

    def train_step(self):
        '''
        Runs one epoch of training on this model.
        '''
        if self.epoch > 0:  # self.loss_function.use_adv:
            self.loss_function.update_param_using_p(self.epoch)  # update the adversarial parameter
        print("[{}] epoch {}".format(self.name, self.epoch))
        print("Adversarial parameter rho - {}".format(self.loss_function.adv_param))
        print("Learning rate - {}".format(self.get_learning_rate()))
        self.model.train()
        # clear the loss
        self.loss = 0.
        self.adv_loss = 0
        # TRAIN
        start_time = time.time()
        print(len(self.dataloader))
        for i_batch, sample_batched in enumerate(self.dataloader):
            print("Batch {} in epoch {} -".format(i_batch, self.epoch))
            # zero gradients before EVERY optimizer step
            self.model.zero_grad()

            pred_info, labels = self.get_pred_with_grad(sample_batched)

            label_tensor = torch.tensor(labels, device=('cuda' if self.use_cuda else 'cpu'))    #Getting stance labels
            topic_tensor = torch.tensor([b['topic_i'] for b in sample_batched],                 #Getting topic indices for train topics
                                        device=('cuda' if self.use_cuda else 'cpu'))
            pred_info['W'] = self.model.trans_layer.W
            pred_info['topic_i'] = topic_tensor         #Assigning topic indices to this dictionary element which is then used to calc adversarial loss on predicting train data topics

            # While training we want to compute adversarial loss.
            graph_loss_all, graph_loss_adv = self.loss_function(pred_info, label_tensor, compute_adv_loss=True)
            self.loss += graph_loss_all.item()
            self.adv_loss += graph_loss_adv.item()
            graph_loss_all.backward(retain_graph=True)  # NOT on adv. params
            # graph_loss_all.backward(retain_graph=self.loss_function.use_adv) # NOT on adv. params
            self.optimizer.step()

            print("Main loss", graph_loss_all.item())

            self.model.zero_grad()
            # if self.loss_function.use_adv:
            if True:  # self.loss_function.use_adv: - always do this, train adversary a bit first on it's own
                print("Adv loss", graph_loss_adv.item())
                graph_loss_adv.backward()
                self.adv_optimizer.step()
                # only on adv params

        end_time = time.time()
        # self.dataloader.reset()
        print("   took: {:.1f} min".format((end_time - start_time) / 60.))
        self.epoch += 1
        self.adjust_learning_rate(self.epoch)                # Adjusts the main and adversary optimizer learning rates using logic in base paper.

    def predict(self, data=None, data_name='DEV'):
        all_y_pred = None
        true_labels = None
        all_top_pred = None
        true_topics = None
        all_marks = None
        all_tar_in_twe = None

        self.model.eval()
        self.loss = 0.
        self.adv_loss = 0.

        if data is None:
            data = self.dataloader

        t2pred = dict()
        for sample_batched in data:
            with torch.no_grad():
                pred_info, labels = self.get_pred_noupdate(sample_batched)

                label_tensor = torch.tensor(labels, device=('cuda' if self.use_cuda else 'cpu'))
                pred_info['W'] = self.model.trans_layer.W

                if data_name == 'TRAIN':        #Predicting on train data the adversarial loss - irrespective of whether adv is included in main model or not
                    topics = [b['topic_i'] for b in sample_batched]
                    topic_tensor = torch.tensor(topics, device=('cuda' if self.use_cuda else 'cpu'))
                    pred_info['topic_i'] = topic_tensor
                    graph_loss_all, graph_loss_adv = self.loss_function(pred_info, label_tensor, compute_adv_loss=True)
                else:
                    # graph_loss_adv will be 0 - not calculated. graph_loss_all won't include adv loss.
                    graph_loss_all, graph_loss_adv = self.loss_function(pred_info, label_tensor, compute_adv_loss=False)

                self.loss += graph_loss_all.item()
                self.adv_loss += graph_loss_adv.item()

                y_pred_arr = pred_info['stance_pred'].detach().cpu().numpy()
                ls = np.array(labels)

                m = [b['seen'] for b in sample_batched]
                tar_in_twe = [b['target_in_tweet'] for b in sample_batched]

                if data_name == 'TRAIN':
                    top_pred_arr = pred_info['adv_pred'].detach().cpu().numpy()
                    tops = np.array(topics)

                for bi, b in enumerate(sample_batched):
                    t = b['ori_topic']
                    t2pred[t] = t2pred.get(t, ([], []))
                    t2pred[t][0].append(y_pred_arr[bi, :])
                    t2pred[t][1].append(ls[bi])

                if all_y_pred is None:
                    all_y_pred = y_pred_arr
                    true_labels = ls
                    all_marks = m
                    all_tar_in_twe = tar_in_twe
                    if data_name == 'TRAIN':
                        all_top_pred = top_pred_arr
                        true_topics = tops
                else:
                    all_y_pred = np.concatenate((all_y_pred, y_pred_arr), 0)
                    true_labels = np.concatenate((true_labels, ls), 0)
                    all_marks = np.concatenate((all_marks, m), 0)
                    all_tar_in_twe = np.concatenate((all_tar_in_twe, tar_in_twe), 0)
                    if data_name == 'TRAIN':
                        all_top_pred = np.concatenate((all_top_pred, top_pred_arr), 0)
                        true_topics = np.concatenate((true_topics, tops), 0)

        for t in t2pred:
            t2pred[t] = (np.argmax(t2pred[t][0], axis=1), t2pred[t][1])

        if None not in all_tar_in_twe:
            all_tar_in_twe = np.array(all_tar_in_twe)
            tar_in_twe_mask = np.column_stack((np.zeros(len(all_tar_in_twe)), np.zeros(len(all_tar_in_twe)), all_tar_in_twe))
            all_y_pred = np.where(tar_in_twe_mask == 1, -np.inf, all_y_pred)

        pred_labels = all_y_pred.argmax(axis=1)
        if data_name == 'TRAIN':
            pred_topics = all_top_pred.argmax(axis=1)
        else:
            pred_topics = None

        return pred_labels, true_labels, t2pred, pred_topics, true_topics, all_marks

    def eval_model(self, data=None, class_wise=False, data_name='DEV'):
        # pred_topics and true_topics will be none while evaluating on dev set
        pred_labels, true_labels, t2pred, pred_topics, true_topics, marks = self.predict(data, data_name)
        self.score(pred_labels, true_labels, class_wise, t2pred, marks)

        # compute score on topic prediction task - used to evaluate adversary performance on train dataset during training
        if data_name == 'TRAIN':
            self.compute_scores(f1_score, true_topics, pred_topics, class_wise, 'topic-f')

        return self.score_dict
