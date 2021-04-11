import torch, sys
import torch.nn as nn
import baseline_model_layers as bml
from transformers import  BertForSequenceClassification


class BiCondLSTMModel(torch.nn.Module):
    '''
    Bidirectional Coniditional Encoding LSTM (Augenstein et al, 2016, EMNLP)
    Single layer bidirectional LSTM where initial states are from the topic encoding.
    Topic is also with a bidirectional LSTM. Prediction done with a single layer FFNN with
    tanh then softmax, to use cross-entropy loss.
    '''

    def __init__(self, hidden_dim, embed_dim, input_dim, drop_prob=0, num_layers=1, num_labels=3,
                 use_cuda=False):
        super(BiCondLSTMModel, self).__init__()
        self.use_cuda = use_cuda
        self.num_labels = num_labels

        self.bilstm = bml.BiCondLSTMLayer(hidden_dim, embed_dim, input_dim, drop_prob, num_layers,
                                      use_cuda=use_cuda)
        self.dropout = nn.Dropout(p=drop_prob)  # so we can have dropouts on last layer
        self.pred_layer = bml.PredictionLayer(input_size=2 * num_layers * hidden_dim,
                                          output_size=self.num_labels,
                                          pred_fn=nn.Tanh(), use_cuda=use_cuda)  # This is BiCond specific


    def forward(self, text, topic, text_l, topic_l):

        text = text.transpose(0, 1)  # (T, B, E)
        topic = topic.transpose(0, 1)  # (C,B,E)

        _, combo_fb_hn, _, _ = self.bilstm(text, topic, topic_l, text_l)

        # dropout
        combo_fb_hn = self.dropout(combo_fb_hn)  # (B, H*N, dir*N_layers)

        y_pred = self.pred_layer(combo_fb_hn)  # (B, 2)
        return y_pred




class AdversarialBasic(torch.nn.Module):
    def __init__(self, enc_params, enc_type, stance_dim, topic_dim, num_labels, num_topics,
                 drop_prob=0.0, use_cuda=False):
        super(AdversarialBasic, self).__init__()
        self.enc_type = enc_type
        self.use_cuda = use_cuda
        self.hidden_dim = enc_params['h']
        self.embed_dim = enc_params['embed_dim']
        self.stance_dim = stance_dim
        self.num_labels = num_labels
        self.num_topics = num_topics

        if self.enc_type == 'bicond':
            self.enc = bml.BiCondLSTMLayer(hidden_dim=self.hidden_dim, embed_dim=self.embed_dim, input_dim=self.embed_dim,
                                           drop_prob=enc_params['drop_prob'], num_layers=1, use_cuda=use_cuda)
            self.att_layer = bml.ScaledDotProductAttention(input_dim=2*self.hidden_dim, use_cuda=self.use_cuda)
        else:
            print("ERROR: invalid encoder type. exiting")
            sys.exit(1)
        self.in_dropout = nn.Dropout(p=drop_prob)
        self.out_dropout = nn.Dropout(p=drop_prob)

        self.recon_layer = bml.ReconstructionLayer(hidden_dim=self.hidden_dim, embed_dim=self.embed_dim,
                                                   use_cuda=self.use_cuda)

        self.topic_recon_layer = bml.ReconstructionLayer(hidden_dim=self.hidden_dim, embed_dim=self.embed_dim, use_cuda=self.use_cuda)
        self.trans_layer = bml.TransformationLayer(input_size=2*self.hidden_dim)

        multiplier = 4
        self.stance_classifier = bml.TwoLayerFFNNLayer(input_dim=multiplier*self.hidden_dim, hidden_dim=stance_dim,
                                                       out_dim=self.num_labels, nonlinear_fn=nn.ReLU())
        self.topic_classifier = bml.TwoLayerFFNNLayer(input_dim=2*self.hidden_dim, hidden_dim=topic_dim,
                                                       out_dim=self.num_topics, nonlinear_fn=nn.ReLU())

    def forward(self, text, topic, text_l, topic_l, text_mask=None, topic_mask=None):
        # text: (B, T, E), topic: (B, C, E), text_l: (B), topic_l: (B), text_mask: (B, T), topic_mask: (B, C)

        # apply dropout on the input
        dropped_text = self.in_dropout(text)

        # encode the text
        if self.enc_type == 'bicond':
            output, _, last_top_hn, topic_output = self.enc(dropped_text.transpose(0, 1),
                                              topic.transpose(0, 1),
                                              topic_l, text_l)
            output = output.transpose(0, 1)     #output represents the token level text encodings of size (B,T,2*H)
            topic_output = topic_output.transpose(0, 1)   #Token levek topic embeddings of size (B, C, 2*H)
            last_top_hn = last_top_hn.transpose(0, 1).reshape(-1, 2*self.hidden_dim)        #(B, 2*H)
            att_vecs = self.att_layer(output, last_top_hn)      #(B, 2H)


        # reconstruct the original embeddings
        recon_embeds = self.recon_layer(output, text_mask) #(B, L, E)
        # reconstruct topic embeddings
        topic_recon_embeds = self.topic_recon_layer(topic_output, topic_mask)

        # transform the representation
        trans_reps = self.trans_layer(att_vecs) #(B, 2H)

        trans_reps = self.out_dropout(trans_reps)  # adding dropout
        last_top_hn = self.out_dropout(last_top_hn)

        # stance prediction
        # added topic input to stance classifier
        stance_input = torch.cat((trans_reps, last_top_hn), 1)      #(B, 4H)
        stance_preds = self.stance_classifier(stance_input)

        # topic prediction
        topic_preds = self.topic_classifier(trans_reps)
        topic_preds_ = self.topic_classifier(trans_reps.detach())

        pred_info = {'text': text, 'text_l': text_l,
                     'topic': topic, 'topic_l': topic_l,
                     'adv_pred': topic_preds, 'adv_pred_':topic_preds_, 'stance_pred': stance_preds,
                     'topic_recon_embeds': topic_recon_embeds, 'recon_embeds': recon_embeds}

        return pred_info


class JointSeqBERTLayer(torch.nn.Module):
    def __init__(self, num_labels=3, use_cuda=False):
        super(JointSeqBERTLayer, self).__init__()

        self.num_labels = num_labels
        self.use_cuda = use_cuda
        self.bert_layer = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        self.dim = 768
        if self.use_cuda:
            self.bert_layer = self.bert_layer.to('cuda')

    def forward(self, **kwargs):
        output = self.bert_layer(input_ids=kwargs['text_topic_batch'].to('cuda'),
                                 token_type_ids=kwargs['token_type_ids'].to('cuda'),
                                 attention_mask=kwargs['attention_mask'].to('cuda'))
        return output[0]


class WordEmbedLayer(torch.nn.Module):
    def __init__(self, vecs, static_embeds=True, use_cuda=False):
        super(WordEmbedLayer, self).__init__()
        vec_tensor = torch.tensor(vecs)

        self.embeds = nn.Embedding.from_pretrained(vec_tensor, freeze=static_embeds)

        self.dim = vecs.shape[1]
        print("Input layer embedding size -  ", self.dim)
        self.vocab_size = float(vecs.shape[0])
        self.use_cuda = use_cuda

    def forward(self, **kwargs):
        embed_args = {'txt_E': self.embeds(kwargs['text']).type(torch.FloatTensor),  # (B, T, E)
                      'top_E': self.embeds(kwargs['topic']).type(torch.FloatTensor)}  # (B, C, E)
        return embed_args