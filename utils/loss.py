import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

class get_loss_cls(torch.nn.Module):
    def __init__(self, alpha=1):
        super(get_loss_cls, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        '''
        :param pred_bd: [B,classes]
        :param target_bd: [B]
        '''
        loss = F.cross_entropy(pred, target)  # pred_clean: [B, n_classes] target_clean: [B]
        return loss

