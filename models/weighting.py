import torch, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from core.util import set_gpu
class AbsWeighting(nn.Module):
    r"""An abstract class for weighting strategies.
    """
    def __init__(self):
        super(AbsWeighting, self).__init__()
        
    def init_param(self):
        r"""Define and initialize some trainable parameters required by specific weighting methods. 
        """
        pass

    def _compute_grad_dim(self):
        self.grad_index = []
        for param in self.get_share_params():
            self.grad_index.append(param.data.numel())
        self.grad_dim = sum(self.grad_index)

    def _grad2vec(self):
        grad = torch.zeros(self.grad_dim)
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                grad[beg:end] = param.grad.data.view(-1)
            count += 1
        return grad

    def _compute_grad(self, losses, mode, rep_grad=False):
        '''
        mode: backward, autograd
        '''
        if not rep_grad:
            grads = set_gpu(torch.zeros(self.task_num, self.grad_dim))
            for tn in range(self.task_num):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                    grads[tn] = self._grad2vec()
                elif mode == 'autograd':
                    grad = list(torch.autograd.grad(losses[tn], self.get_share_params(), retain_graph=True))
                    grads[tn] = torch.cat([g.view(-1) for g in grad])
                else:
                    raise ValueError('No support {} mode for gradient computation')
                self.zero_grad_share_params()
        else:
            if not isinstance(self.rep, dict):
                grads = set_gpu(torch.zeros(self.task_num, *self.rep.size()))
            else:
                grads = [torch.zeros(*self.rep[task].size()) for task in self.task_name]
            for tn, task in enumerate(self.task_name):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                    grads[tn] = self.rep_tasks[task].grad.data.clone()
        return grads

    def _reset_grad(self, new_grads):
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
            count += 1
            
    def _get_grads(self, losses, mode='backward'):
        r"""This function is used to return the gradients of representations or shared parameters.
        If ``rep_grad`` is ``True``, it returns a list with two elements. The first element is \
        the gradients of the representations with the size of [task_num, batch_size, rep_size]. \
        The second element is the resized gradients with size of [task_num, -1], which means \
        the gradient of each task is resized as a vector.
        If ``rep_grad`` is ``False``, it returns the gradients of the shared parameters with size \
        of [task_num, -1], which means the gradient of each task is resized as a vector.
        """
        if self.rep_grad:
            per_grads = self._compute_grad(losses, mode, rep_grad=True)
            if not isinstance(self.rep, dict):
                grads = per_grads.reshape(self.task_num, self.rep.size()[0], -1).sum(1)
            else:
                try:
                    grads = torch.stack(per_grads).sum(1).view(self.task_num, -1)
                except:
                    raise ValueError('The representation dimensions of different tasks must be consistent')
            return [per_grads, grads]
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode)
            return grads
        
    def _backward_new_grads(self, batch_weight, per_grads=None, grads=None):
        r"""This function is used to reset the gradients and make a backward.
        Args:
            batch_weight (torch.Tensor): A tensor with size of [task_num].
            per_grad (torch.Tensor): It is needed if ``rep_grad`` is True. The gradients of the representations.
            grads (torch.Tensor): It is needed if ``rep_grad`` is False. The gradients of the shared parameters. 
        """
        if self.rep_grad:
            if not isinstance(self.rep, dict):
                transformed_grad = torch.einsum('i, i... -> ...', batch_weight, per_grads)
                self.rep.backward(transformed_grad)
            else:
                for tn, task in enumerate(self.task_name):
                    rg = True if (tn+1)!=self.task_num else False
                    self.rep[task].backward(batch_weight[tn]*per_grads[tn], retain_graph=rg)
        else:
            new_grads = torch.einsum('i, i... -> ...', batch_weight, grads)
            self._reset_grad(new_grads)
    
    @property
    def backward(self, losses, **kwargs):
        r"""
        Args:
            losses (list): A list of losses of each task.
            kwargs (dict): A dictionary of hyperparameters of weighting methods.
        """
        pass

class GradVac(AbsWeighting):
    r"""Gradient Vaccine (GradVac).
    
    Args:
        beta (float, default=0.5): The exponential moving average (EMA) decay parameter.
    .. warning::
            GradVac is not supported by representation gradients, i.e., ``rep_grad`` must be ``False``.
    """
    def __init__(self, num_loss):
        super(GradVac, self).__init__()
        self.task_num = num_loss
    def init_param(self):
        self.rho_T = set_gpu(torch.zeros(self.task_num, self.task_num))
        
    def backward(self, losses, **kwargs):
        beta = 0.5

        self.init_param()
        self._compute_grad_dim()
        grads = self._compute_grad(losses, mode='backward') # [task_num, grad_dim]
        batch_weight = np.ones(len(losses))
        pc_grads = grads.clone()
        for tn_i in range(self.task_num):
            task_index = list(range(self.task_num))
            task_index.remove(tn_i)
            random.shuffle(task_index)
            for tn_j in task_index:
                rho_ij = torch.dot(pc_grads[tn_i], grads[tn_j]) / (pc_grads[tn_i].norm()*grads[tn_j].norm())
                if rho_ij < self.rho_T[tn_i, tn_j]:
                    w = pc_grads[tn_i].norm()*(self.rho_T[tn_i, tn_j]*(1-rho_ij**2).sqrt()-rho_ij*(1-self.rho_T[tn_i, tn_j]**2).sqrt())/(grads[tn_j].norm()*(1-self.rho_T[tn_i, tn_j]**2).sqrt())
                    pc_grads[tn_i] += grads[tn_j]*w
                    batch_weight[tn_j] += w.item()
                    self.rho_T[tn_i, tn_j] = (1-beta)*self.rho_T[tn_i, tn_j] + beta*rho_ij
        new_grads = pc_grads.sum(0)
        self._reset_grad(new_grads)
        return batch_weight
    def get_share_params(self):
        return self.net.parameters()
    def zero_grad_share_params(self):
        self.net.zero_grad()

class GradNorm(AbsWeighting):
    r"""Gradient Normalization (GradNorm).
    
    Args:
        alpha (float, default=1.5): The strength of the restoring force which pulls tasks back to a common training rate.
    """
    def __init__(self, num_loss):
        super(GradNorm, self).__init__()
        self.task_num = num_loss
        self.rep_grad = False
    def init_param(self):
        self.loss_scale = set_gpu(nn.Parameter(torch.tensor([1.0]*self.task_num)))
    def backward(self, losses, epoch, **kwargs):
        alpha = 1.5
        self.init_param()

        if epoch >= 1:
            loss_scale = self.task_num * F.softmax(self.loss_scale, dim=-1)
            grads = self._get_grads(losses, mode='backward')
            if self.rep_grad:
                per_grads, grads = grads[0], grads[1]
                
            G_per_loss = torch.norm(loss_scale.unsqueeze(1)*grads, p=2, dim=-1)
            G = G_per_loss.mean(0)
            L_i = set_gpu(torch.Tensor([losses[tn].item()/self.train_loss_buffer[tn] for tn in range(self.task_num)]))
            r_i = L_i/L_i.mean()
            constant_term = (G*(r_i**alpha)).detach()
            L_grad = (G_per_loss-constant_term).abs().sum(0)
            L_grad.backward()
            loss_weight = loss_scale.detach().clone()
            
            if self.rep_grad:
                self._backward_new_grads(loss_weight, per_grads=per_grads)
            else:
                self._backward_new_grads(loss_weight, grads=grads)
            return loss_weight.cpu().numpy()
        else:
            self.train_loss_buffer = losses.detach().cpu().numpy()
            loss = torch.mul(losses, set_gpu(torch.ones_like(losses))).sum()
            loss.backward()
            return np.ones(self.task_num)
    def get_share_params(self):
        return self.net.parameters()
    def zero_grad_share_params(self):
        self.net.zero_grad()