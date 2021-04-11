import torch, pickle, json
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import BertTokenizer
import pandas as pd
from functools import reduce


class StanceData(Dataset):
    '''
    Holds the stance dataset.
    '''
    def __init__(self, data_name, vocab_name, topic_name=None, name='',
                 max_sen_len=10, max_tok_len=200, max_top_len=5, binary=False,
                 pad_val=0, is_bert=False, add_special_tokens=True, use_tar_in_twe=False):
        self.data_name = data_name
        self.data_file = pd.read_csv(data_name)
        if vocab_name is not None:
            self.word2i = pickle.load(open(vocab_name, 'rb'))
        self.name = name
        self.max_sen_len = max_sen_len
        self.max_tok_len = max_tok_len
        self.max_top_len = max_top_len
        self.binary = binary
        self.pad_value = pad_val
        self.topic2i = pickle.load(open(topic_name, 'rb')) if topic_name is not None else dict()
        self.is_bert = is_bert
        self.add_special_tokens = add_special_tokens
        self.tar_in_twe = ('target_in_tweet' in self.data_file.columns)
        self.use_tar_in_twe = use_tar_in_twe

        if self.is_bert:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.preprocess_data()

        if self.is_bert:
            # filter unlabeled examples for Twitter
            self.data_file = self.data_file.loc[self.data_file['label'] != 3]
            self.data_file.reset_index(inplace=True)  # reset the index so we can access correctly later

    def process_bert(self):
        print("processing BERT")
        topic_str_lst = []
        text_str_lst = []
        for i in self.data_file.index:
            row = self.data_file.iloc[i]
            num_sens = 1
            if 'topic_str' in self.data_file.columns:
                ori_topic = row['topic_str']
            else:
                ori_topic = ' '.join(json.loads(row['topic']))
                topic_str_lst.append(ori_topic)

            if 'text_s' in self.data_file.columns:
                ori_text = row['text_s']
            else:
                ori_text = ' '.join(sum(json.loads(row['text']), []))
                text_str_lst.append(ori_text)

            text_topic = self.tokenizer(ori_text, ori_topic, padding='max_length', max_length=int(self.max_tok_len),
                                        return_token_type_ids=True,
                                        return_attention_mask=True)
            text = self.tokenizer(ori_text, add_special_tokens=self.add_special_tokens,
                                  max_length=int(self.max_tok_len), padding='max_length')
            topic = self.tokenizer(ori_topic, add_special_tokens=self.add_special_tokens,
                                   max_length=int(self.max_top_len), padding='max_length')
            self.data_file.at[i, 'text_idx'] = text['input_ids']
            self.data_file.at[i, 'ori_text'] = ori_text
            self.data_file.at[i, 'topic_idx'] = topic['input_ids']
            self.data_file.at[i, 'num_sens'] = num_sens
            self.data_file.at[i, 'text_topic_idx'] = text_topic['input_ids']
            self.data_file.at[i, 'token_type_ids'] = text_topic['token_type_ids']
            self.data_file.at[i, 'attention_mask'] = text_topic['attention_mask']
        print("...finished pre-processing for BERT")
        if 'topic_str' not in self.data_file.columns:
            self.data_file['topic_str'] = topic_str_lst

        if 'text_s' not in self.data_file.columns:
            self.data_file['text_s'] = text_str_lst
        return


    def process_nonbert(self):
        # Creating topic_string from tokenized topic column for twitter dataset
        if 'topic_str' not in self.data_file.columns:
            add_topic_string = True
        else:
            add_topic_string = False

        for i in self.data_file.index:
            row = self.data_file.iloc[i]

            # Tokenized text in the form of [[tokenized sentence 1],[tokenized sent 2],...].
            # In twitter data it is a 2 D array with [[tokenized text]].
            ori_text = json.loads(row['text'])
            # Tokenized topic array - 1D array with tokenized topic
            ori_topic = json.loads(row['topic'])

            # index text & topic
            text = [[self.get_index(w) for w in s] for s in ori_text]
            topic = [self.get_index(w) for w in ori_topic][:self.max_top_len]

            text = reduce(lambda x, y: x + y, text)
            text = text[:self.max_tok_len]
            text_lens = len(text)  # compute combined text len
            num_sens = 1
            text_mask = [1] * text_lens

            while len(text) < self.max_tok_len:
                text.append(self.pad_value)
                text_mask.append(0)

            # compute topic len
            topic_lens = len(topic)  # get len (before padding)
            topic_mask = [1] * topic_lens

            # pad topic
            while len(topic) < self.max_top_len:
                topic.append(self.pad_value)
                topic_mask.append(0)

            if 'text_s' in self.data_file.columns:
                ori_text_ = row['text_s']
            else:
                ori_text_ = ' '.join([' '.join(ti) for ti in ori_text])

            if add_topic_string:
                self.data_file.at[i, 'topic_str'] = ' '.join(ori_topic)

            self.data_file.at[i, 'text_idx'] = text
            self.data_file.at[i, 'topic_idx'] = topic
            self.data_file.at[i, 'text_l'] = text_lens
            self.data_file.at[i, 'topic_l'] = topic_lens
            self.data_file.at[i, 'ori_text'] = ori_text_
            self.data_file.at[i, 'num_sens'] = num_sens
            self.data_file.at[i, 'text_mask'] = text_mask
            self.data_file.at[i, 'topic_mask'] = topic_mask

    def preprocess_data(self):
        print('preprocessing data {} ...'.format(self.data_name))

        self.data_file['text_idx'] = [[] for _ in range(len(self.data_file))]
        self.data_file['topic_idx'] = [[] for _ in range(len(self.data_file))]
        self.data_file['text_topic_idx'] = [[] for _ in range(len(self.data_file))]
        self.data_file['token_type_ids'] = [[] for _ in range(len(self.data_file))]
        self.data_file['text_l'] = 0
        self.data_file['ori_text'] = ''
        self.data_file['topic_l'] = 0
        self.data_file['num_sens'] = 0
        self.data_file['text_mask'] = [[] for _ in range(len(self.data_file))]
        self.data_file['topic_mask'] = [[] for _ in range(len(self.data_file))]

        if self.is_bert:
            self.process_bert()
        else:
            self.process_nonbert()

        print("... finished preprocessing")

    def get_index(self, word):
        return self.word2i[word] if word in self.word2i else len(self.word2i)

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx, corpus=None):
        row = self.data_file.iloc[idx]

        l  = int(row['label'])

        if self.tar_in_twe and self.use_tar_in_twe:
            tar_in_twe_value = row['target_in_tweet']
        else:
            tar_in_twe_value = None

        sample = {'text': row['text_idx'], 'topic': row['topic_idx'],
                  'label': l,
                  'txt_l': row['text_l'], 'top_l': row['topic_l'],
                  'ori_topic': row['topic_str'],
                  'ori_text': row['ori_text'],
                  'text_mask': row['text_mask'],
                  'num_s': row['num_sens'],
                  'seen': row['seen?'],
                  }
        if self.is_bert and not self.add_special_tokens:
            sample['text_topic'] = row['text_topic_idx']
            sample['token_type_ids'] = row['token_type_ids']
            sample['attention_mask'] = row['attention_mask']
        else:
            sample['topic_i'] = self.topic2i.get(row['topic'], 0)
            sample['topic_mask'] = row['topic_mask']
            sample['target_in_tweet'] = tar_in_twe_value

        return sample