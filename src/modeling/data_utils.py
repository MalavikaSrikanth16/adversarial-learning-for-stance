import torch, random
import numpy as np


def load_vectors(vecfile, dim=300, unk_rand=True, seed=0):
    '''
    Loads saved vectors;
    :param vecfile: the name of the file to load the vectors from.
    :return: a numpy array of all the vectors.
    '''
    vecs = np.load(vecfile)
    np.random.seed(seed)

    if unk_rand:
        vecs = np.vstack((vecs, np.random.randn(dim))) # <unk> -> V-2
    else:
        vecs = np.vstack((vecs, np.zeros(dim))) # <unk> -> V - 2
    vecs = np.vstack((vecs, np.zeros(dim))) # pad -> V-1
    vecs = vecs.astype(float, copy=False)

    return vecs

def prepare_batch(sample_batched, **kwargs):
    '''
    Prepares a batch of data to be used in training or evaluation. Includes the text reversed.
    :param sample_batched: a list of dictionaries, where each is a sample
    :return: a dictionary containing:
            a tensor of all the text instances,
            a tensor of all topic instances,
            a list of labels for the text,topic instances
            a list of the text lengths
            a list of the topic lengths
            a list with the original texts
            a list with the original topics
            AND (depending on flags)
            a tensor of the inputs in the format CLS text SEP topic SEP (for Bert)
            a tensor of the token type ids (for Bert)
            a tensor with the generalized topic representations
    '''
    text_lens = np.array([b['txt_l'] for b in sample_batched])
    topic_batch = torch.tensor([b['topic'] for b in sample_batched])
    labels = [b['label'] for b in sample_batched]
    top_lens = [b['top_l'] for b in sample_batched]

    raw_text_batch = [b['ori_text'] for b in sample_batched]
    raw_top_batch = [b['ori_topic'] for b in sample_batched]

    text_batch = torch.tensor([b['text'] for b in sample_batched])

    args = {'text': text_batch, 'topic': topic_batch, 'labels': labels,
            'txt_l': text_lens, 'top_l': top_lens,
            'ori_text': raw_text_batch, 'ori_topic': raw_top_batch}

    if 'text_topic' in sample_batched[0]:
        args['text_topic_batch'] = torch.tensor([b['text_topic'] for b in sample_batched])
        args['token_type_ids'] = torch.tensor([b['token_type_ids'] for b in sample_batched])
        args['attention_mask'] = torch.tensor([b['attention_mask'] for b in sample_batched])

    if 'topic_rep_id' in sample_batched[0]:
        args['topic_rep_ids'] = torch.tensor([b['topic_rep_id'] for b in sample_batched])

    return args


def prepare_batch_adv(sample_batched, **kwargs):
    args = prepare_batch(sample_batched, **kwargs)

    txt_mask = [b['text_mask'] for b in sample_batched]
    args['txt_mask'] = txt_mask

    top_mask = [b['topic_mask'] for b in sample_batched]
    args['top_mask'] = top_mask

    return args


class DataSampler:
    '''
    A sampler for a dataset. Can get samples of differents sizes.
    Is iterable. By default shuffles the data each time all the data
    has been used through iteration.
    '''
    def __init__(self, data, batch_size, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        random.seed(0)

        self.indices = list(range(len(data)))
        if shuffle:
            random.shuffle(self.indices)
        self.batch_num = 0

    def __len__(self):
        return len(self.data)

    def num_batches(self):
        return len(self.data) / float(self.batch_size)

    def __iter__(self):
        self.indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.indices != []:
            idxs = self.indices[:self.batch_size]
            batch = [self.data.__getitem__(i) for i in idxs]
            self.indices = self.indices[self.batch_size:]
            return batch
        else:
            raise StopIteration

    def get(self):
        self.reset()
        return self.__next__()

    def reset(self):
        self.indices = list(range(len(self.data)))
        if self.shuffle: random.shuffle(self.indices)


def setup_helper_bicond(args, use_cuda):
    if use_cuda:
        txt_E= args['txt_E'].to('cuda')  # (B,T,E)
        top_E = args['top_E'].to('cuda')  # (B,C,E)
        txt_l = torch.tensor(args['txt_l']).to('cuda')  # (B, S)
        top_l = torch.tensor(args['top_l']).to('cuda')  # (B)
    else:
        txt_E = args['txt_E']  # (B,T,E)
        top_E = args['top_E']  # (B,C,E)
        txt_l = torch.tensor(args['txt_l'])
        top_l = torch.tensor(args['top_l'])
    return txt_E, top_E, txt_l, top_l


def setup_helper_adv(args, use_cuda):
    if use_cuda:
        txt_E= args['txt_E'].to('cuda')  # (B,T,E)
        top_E = args['top_E'].to('cuda')  # (B,C,E)
    else:
        txt_E = args['txt_E']  # (B,T,E)
        top_E = args['top_E']  # (B,C,E)

    device = 'cuda' if use_cuda else 'cpu'

    txt_l = torch.tensor(args['txt_l'], device=device)  # (B, S)
    top_l = torch.tensor(args['top_l'], device=device) # (B)
    txt_mask = torch.tensor(args['txt_mask'], device=device) # (B, T)
    top_mask = torch.tensor(args['top_mask'], device=device) # (B, C)

    return txt_E, top_E, txt_l, top_l, txt_mask, top_mask
