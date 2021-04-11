import torch
import torch.nn as nn
import math
from IPython import embed

class ReconstructionLoss(torch.nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, ori_embeds, model_embeds, embed_l):
        # (B, L, E)
        temp = torch.norm(model_embeds - self.tanh(ori_embeds), dim=2) ** 2
        lrec = temp.sum(1) / embed_l
        return lrec.mean() # combine the loss across the batch


class TransformationLoss(torch.nn.Module):
    def __init__(self, dim, l, use_cuda=False):
        super(TransformationLoss, self).__init__()

        self.eye = torch.eye(dim, device='cuda' if use_cuda else 'cpu')
        self.l = l

    def forward(self, W):
        temp =  self.l * torch.norm(W - self.eye) ** 2
        return temp


class AdvBasicLoss(torch.nn.Module):
    def __init__(self, trans_dim, trans_param, num_no_adv=None, tot_epochs=20, rho_adv=False, gamma=10,
                 rec_weight=1, semi_sup=False, use_cuda=False):
        super(AdvBasicLoss, self).__init__()

        self.rec_loss = ReconstructionLoss()
        self.trans_loss = TransformationLoss(dim=trans_dim, l=trans_param, use_cuda=use_cuda)

        self.adv_param = 0. # start with the adversary weight set to 0


        self.semi_sup = semi_sup
        if self.semi_sup:
            self.stance_loss = nn.CrossEntropyLoss(ignore_index=3)
        else:
            self.stance_loss = nn.CrossEntropyLoss()
        self.topic_loss = nn.CrossEntropyLoss()
        #Adversary is not used for num_no_adv initial epochs
        self.use_adv = num_no_adv == 0
        self.num_no_adv = num_no_adv
        self.tot_epochs = tot_epochs
        self.rec_weight = rec_weight
        self.i = 0
        self.rho_adv = rho_adv
        self.gamma = gamma
        self.use_cuda = use_cuda

    def update_param_using_p(self, epoch):
        if epoch >= self.num_no_adv:
            self.use_adv = True
            tot_epochs_for_calc = self.tot_epochs - self.num_no_adv
            epoch_for_calc = epoch - self.num_no_adv
            p = epoch_for_calc/tot_epochs_for_calc

            self.adv_param = 2/(1 + math.exp(-self.gamma*p)) - 1
        else:
            self.use_adv = False

    def forward(self, pred_info, labels, compute_adv_loss=True, print_=False):
        lrec = self.rec_weight * self.rec_loss(ori_embeds=pred_info['text'], model_embeds=pred_info['recon_embeds'],
                         embed_l=pred_info['text_l'])
        lrec_topic = self.rec_weight * self.rec_loss(ori_embeds=pred_info['topic'], model_embeds=pred_info['topic_recon_embeds'],
                                                 embed_l=pred_info['topic_l'])

        ltrans = self.trans_loss(W=pred_info['W'])
        llabel = self.stance_loss(pred_info['stance_pred'], labels)
        ladv = torch.tensor(0)
        adversarial_loss = torch.tensor(0)
        if self.use_cuda:
            ladv = ladv.to('cuda')
            adversarial_loss = adversarial_loss.to('cuda')
        if compute_adv_loss:        #Ladv is computed only on the train dataset else it is left as 0.
            ladv = self.topic_loss(pred_info['adv_pred'], pred_info['topic_i'])
            if self.rho_adv:
                adversarial_loss = self.adv_param * self.topic_loss(pred_info['adv_pred_'], pred_info['topic_i'])
            else:
                adversarial_loss = self.topic_loss(pred_info['adv_pred_'], pred_info['topic_i'])

        if print_:
            print("lrec - {}, lrec_topic - {}, ltrans - {}, llabel - {}, ladv - {}".format(lrec, lrec_topic, ltrans, llabel, ladv))

        self.i += 1
        if self.use_adv:
            if self.i % 100 == 0:
                print("loss: {:.4f} + {:.4f} + {:.4f} - {:.4f}; adv: {:.4f}".format(lrec.item(), ltrans.item(), llabel.item(),
                                                   (self.adv_param * ladv).item(), ladv))
            return lrec + lrec_topic + ltrans + llabel - self.adv_param * ladv, adversarial_loss
        else:
            if self.i % 100 == 0:
                print("loss: {:.4f} +  {:.4f} + {:.4f}; adv: {:.4f}".format(lrec.item(), ltrans.item(), llabel.item(),
                                                     ladv))
            return lrec + lrec_topic + ltrans + llabel, adversarial_loss
