import torch
import torch.nn as nn


#类别加权
def CW_loss(class_weights):
    return nn.CrossEntropyLoss(weight=class_weights)


#焦点损失
class FocalLoss(nn.Module):
    def __init__(self,alpha=None,gamma=2.0,reduction='mean'):
        super().__init__()
        self.gamma=gamma
        self.alpha=alpha
        self.reduction=reduction

    def forward(self,inputs,targets):
        ce_loss=nn.CrossEntropyLoss(inputs,targets,reduction='none')
        p_true=torch.exp(-ce_loss)
        focal_loss=(1-p_true)**self.gamma*ce_loss
        
        if self.alpha is not None:
            alpha_t=self.alpha[targets]
            focal_loss *=alpha_t
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

