import torch, math
import torch.nn as nn
import torch.nn.utils.rnn as rnn


class TwoLayerFFNNLayer(torch.nn.Module):
    '''
    2-layer FFNN with specified nonlinear function
    must be followed with some kind of prediction layer for actual prediction
    '''
    def __init__(self, input_dim, hidden_dim, out_dim, nonlinear_fn):
        super(TwoLayerFFNNLayer, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.model = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                   nonlinear_fn,
                                   nn.Linear(hidden_dim, out_dim))

    def forward(self, input):
        return self.model(input)


class PredictionLayer(torch.nn.Module):
    '''
    Predicition layer. linear projection followed by the specified functions
    ex: pass pred_fn=nn.Tanh()
    '''
    def __init__(self, input_size, output_size, pred_fn, use_cuda=False):
        super(PredictionLayer, self).__init__()

        self.use_cuda = use_cuda

        self.input_dim = input_size
        self.output_dim = output_size
        self.pred_fn = pred_fn

        self.model = nn.Sequential(nn.Linear(self.input_dim, self.output_dim, bias=False))

        if self.use_cuda:
            self.model = self.model.to('cuda')#cuda()

    def forward(self, input_data):
        return self.model(input_data)


class BiCondLSTMLayer(torch.nn.Module):
    '''
    Bidirection Conditional Encoding (Augenstein et al. 2016 EMNLP).
    Bidirectional LSTM with initial states from topic encoding.
    Topic encoding is also a bidirectional LSTM.
    '''
    def __init__(self, hidden_dim, embed_dim, input_dim, drop_prob=0, num_layers=1, use_cuda=False):
        super(BiCondLSTMLayer, self).__init__()

        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.use_cuda = use_cuda

        self.topic_lstm = nn.LSTM(input_dim, self.hidden_dim, bidirectional=True)
        self.text_lstm = nn.LSTM(self.embed_dim, self.hidden_dim, bidirectional=True)

    def forward(self, txt_e, top_e, top_l, txt_l):
        ####################
        # txt_e = (Lx, B, E), top_e = (Lt, B, E), top_l=(B), txt_l=(B)
        ########################
        p_top_embeds = rnn.pack_padded_sequence(top_e, top_l, enforce_sorted=False)

        self.topic_lstm.flatten_parameters()

        # feed topic
        topic_output, last_top_hn_cn = self.topic_lstm(p_top_embeds)  # (seq_ln, B, 2*H),((2, B, H), (2, B, H))
        last_top_hn = last_top_hn_cn[0]  # LSTM
        padded_topic_output, _ = rnn.pad_packed_sequence(topic_output,total_length=top_e.shape[0])

        p_text_embeds = rnn.pack_padded_sequence(txt_e, txt_l, enforce_sorted=False)
        self.text_lstm.flatten_parameters()

        # feed text conditioned on topic
        output, (txt_last_hn, _)  = self.text_lstm(p_text_embeds, last_top_hn_cn) # (2, B, H)
        txt_fw_bw_hn = txt_last_hn.transpose(0, 1).reshape((-1, 2 * self.hidden_dim))
        padded_output, _ = rnn.pad_packed_sequence(output, total_length=txt_e.shape[0])
        return padded_output, txt_fw_bw_hn, last_top_hn, padded_topic_output


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, input_dim, use_cuda=False):
        super(ScaledDotProductAttention, self).__init__()
        self.input_dim = input_dim

        self.scale = math.sqrt(2 * self.input_dim)

    def forward(self, inputs, query):
        # inputs = (B, L, 2*H), query = (B, 2*H), last_hidden=(B, 2*H)
        sim = torch.einsum('blh,bh->bl', inputs, query) / self.scale  # (B, L)
        att_weights = nn.functional.softmax(sim, dim=1)  # (B, L)
        context_vec = torch.einsum('blh,bl->bh', inputs, att_weights)  # (B, 2*H)
        return context_vec
    

class TransformationLayer(torch.nn.Module):
    '''
    Linear transformation layer
    '''
    def __init__(self, input_size):
        super(TransformationLayer, self).__init__()

        self.dim = input_size

        self.W = torch.empty((self.dim, self.dim))
        self.W = nn.Parameter(nn.init.xavier_normal_(self.W)) # (D, D)

    def forward(self, text):
        # text: (B, D)
        return torch.einsum('bd,dd->bd', text, self.W)


class ReconstructionLayer(torch.nn.Module):
    '''
    Embedding reconstruction layer
    '''
    def __init__(self, hidden_dim, embed_dim, use_cuda=False):
        super(ReconstructionLayer, self).__init__()

        self.hidden_dim = hidden_dim
        self.embed_dim=embed_dim
        self.use_cuda = use_cuda

        self.recon_W = torch.empty((2 * self.hidden_dim, self.embed_dim),
                                   device=('cuda' if self.use_cuda else 'cpu'))
        self.recon_w = nn.Parameter(nn.init.xavier_normal_(self.recon_W))
        self.recon_b = torch.empty((self.embed_dim, 1), device=('cuda' if self.use_cuda else 'cpu'))
        self.recon_b = nn.Parameter(nn.init.xavier_normal_(self.recon_b)).squeeze(1)
        self.tanh = nn.Tanh()

    def forward(self, text_output, text_mask):
        # text_output: (B, T, H), text_mask: (B, T)
        recon_embeds = self.tanh(torch.einsum('blh,he->ble', text_output, self.recon_w) + self.recon_b)  # (B,L,E)
        recon_embeds = torch.einsum('ble,bl->ble', recon_embeds, text_mask)

        return recon_embeds