import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from core.util import GradLayer
from core.util import set_gpu
# class mse_loss(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.loss_fn = nn.MSELoss()
#     def forward(self, output, target):
#         return self.loss_fn(output, target)


def mse_loss(output, target):
    return F.mse_loss(output, target)

# def inner_loss(output, target, clean_cond):
#     loss = torch.square(output - target)
#     weight = torch.exp(torch.abs(target - clean_cond))
#     return torch.mean(loss * weight) + mse_loss(output, target)

def inner_loss(output, target, cond):
    loss = torch.square(output - target)
    weight = torch.exp(torch.abs(target - cond))
    return torch.mean(loss * weight) + mse_loss(output, target)

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.edge_loss = EdgeLoss()
    def forward(self, noise_hat, noise, y_0_hat, target, attention_hat, attention):
        out_loss = torch.square(y_0_hat - target)
        # noise_loss = torch.square(noise - noise_hat)
        # weight = torch.exp(torch.abs(target - cond))
        attention_loss = torch.square(attention_hat - attention)
        return  self.edge_loss(y_0_hat, target)+\
                mse_loss(noise, noise_hat)+\
                torch.mean(out_loss * attention)+\
                torch.mean(attention_loss)
                
    def __call__(self,noise_hat, noise, output, target, attention_hat, attention):
        return self.forward(noise_hat, noise, output, target, attention_hat, attention)

class EdgeLoss(nn.Module):

    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.loss = F.mse_loss
        self.grad_layer = GradLayer()

    def forward(self, output, target):
        output_grad = self.grad_layer(output)
        gt_grad = self.grad_layer(target)
        return self.loss(output_grad, gt_grad)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
    def __call__(self, input, target):
        return self.forward(input, target)
