import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def stable_dataloader(dataloader,seed,epoch):
    seed+=epoch
    set_seed(seed)
    return dataloader

def unitwise_norm(x, norm_type=2.0):
    if x.ndim <= 1:
        return x.norm(norm_type)
    else:
        return x.norm(norm_type, dim=tuple(range(1, x.ndim))).unsqueeze(1)
def adaptive_clip_grad(parameters, clip_factor=0.01, eps=1e-3, norm_type=2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in parameters:
        if p.grad is None:
            continue
        p_data = p.detach()
        g_data = p.grad.detach()
        max_norm = unitwise_norm(p_data, norm_type=norm_type).clamp_(min=eps).mul_(clip_factor)
        grad_norm = unitwise_norm(g_data, norm_type=norm_type)
        clipped_grad = g_data * (max_norm / grad_norm.clamp(min=1e-6))
        new_grads = torch.where(grad_norm < max_norm, g_data, clipped_grad)
        p.grad.detach().copy_(new_grads)
    
class View(nn.Module):

    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

def cuda(tensor, device):

    if device == -1:
        return tensor
    return tensor.cuda(device)



class ImprovedDeepSurvLoss(nn.Module):
    def __init__(self, alpha=0.0):
        super(ImprovedDeepSurvLoss, self).__init__()
        self.alpha = alpha

    def forward(self, risk_pred, durations, events):
        mask = torch.ones(durations.shape[0], durations.shape[0], device=risk_pred.device)
        mask[(durations.T - durations) > 0] = 0

        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss + 1e-7).reshape(-1, 1)

        neg_log_loss = -torch.sum((risk_pred - log_loss) * events) / (torch.sum(events) + 1e-7)
        
        l2_reg = torch.mean(risk_pred ** 2)
        
        loss = neg_log_loss + self.alpha * l2_reg
        
        return loss
    

class ImprovedDeepSurvLoss2(nn.Module):
    def __init__(self, eps=1e-7):
        super(ImprovedDeepSurvLoss2, self).__init__()
        self.eps = eps
        
    def forward(self, risk_pred, durations, events):
        mask = (durations[:, None] - durations[None, :] > 0) & (events[None, :] > 0)
        risk_pred_exp = torch.exp(risk_pred)
        log_loss = torch.sum(risk_pred_exp * mask, dim=1) / (torch.sum(mask, dim=1) + self.eps)
        neg_log_loss = -torch.sum((risk_pred - log_loss.reshape(-1, 1)) * events) / (torch.sum(events) + self.eps)
        return neg_log_loss


class PriorMultiLabelSoftMarginLoss(nn.Module):
    def __init__(self, prior=None, num_labels=None, reduction="mean", eps=1e-9, tau=1.0, device="cuda"):
        super(PriorMultiLabelSoftMarginLoss, self).__init__()
        self.loss_mlsm = torch.nn.MultiLabelSoftMarginLoss(reduction=reduction)
        
        if not prior:
            if num_labels is None:
                raise ValueError("num_labels must be provided when prior is not given")
            prior = np.array([1 / num_labels for _ in range(num_labels)])
        
        if type(prior) == list:
            prior = np.array(prior)
        
        self.log_prior = torch.tensor(np.log(prior + eps), device=device).unsqueeze(0)
        self.eps = eps
        self.tau = tau
        self.device = device

    def forward(self, logits, labels):
        labels_2d = torch.zeros_like(logits, device=self.device)
        labels = labels.long()
        labels_2d[torch.arange(labels.size(0)), labels] = 1
        
        logits = logits + self.tau * self.log_prior
        loss = self.loss_mlsm(logits, labels_2d)
        
        return loss
    

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights=None, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = weights
        self.reduction = reduction

    def forward(self, logits, labels):
        if self.weights is not None:
            self.weights = self.weights.to(logits.device)
            log_softmax = nn.LogSoftmax(dim=1)
            log_probs = log_softmax(logits)
            loss = torch.sum(-self.weights[labels] * log_probs[torch.arange(labels.size(0)), labels])
        else:
            loss = nn.functional.cross_entropy(logits, labels, reduction=self.reduction)
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss

class MaxDiversityConsistencyLoss(nn.Module):
    def __init__(self, diversity_weight=1.0, consistency_weight=1.0):
        super(MaxDiversityConsistencyLoss, self).__init__()
        self.diversity_weight = diversity_weight
        self.consistency_weight = consistency_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, logits_list, ensemble_logits, labels):
        diversity_loss = 0
        for i in range(len(logits_list)):
            for j in range(i+1, len(logits_list)):
                diversity_loss += torch.exp(-self.mse_loss(logits_list[i], logits_list[j]))
        diversity_loss /= (len(logits_list) * (len(logits_list) - 1) / 2)
        diversity_loss = -torch.log(diversity_loss + 1e-8)

        consistency_loss = 0
        for logits in logits_list:
            consistency_loss += self.mse_loss(logits, ensemble_logits)
        consistency_loss /= len(logits_list)


        total_loss =  self.diversity_weight * diversity_loss + self.consistency_weight * consistency_loss

        return total_loss
    
class DeepSurvLoss(nn.Module):
    def __init__(self):
        super(DeepSurvLoss, self).__init__()
    def forward(self, risk_pred, durations, events):
        mask = torch.ones(durations.shape[0], durations.shape[0],device=device)
        mask[(durations.T - durations) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred - log_loss) * events) / torch.sum(events)
        return neg_log_loss

class RankLoss(nn.Module):
    def __init__(self):
        super(RankLoss, self).__init__()
    def forward(self, predictions, durations, events, loss_type='hinge'):
        mask = (durations.unsqueeze(1) < durations.unsqueeze(0)) * (events.unsqueeze(1) > 0)
        diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)

        if loss_type == 'hinge':
            loss = torch.mean(mask * torch.clamp(1 - diff, min=0))
        elif loss_type == 'exp':
            loss = torch.mean(mask * torch.exp(-diff))
        elif loss_type == 'sigmoid':
            loss = torch.mean(mask * torch.sigmoid(diff))
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        return loss
    
class CoxPHLoss(nn.Module):
    def __init__(self):
        super(CoxPHLoss, self).__init__()
    
    def forward(self, predictions, durations, events):
        _, sorted_idx = torch.sort(durations,descending=False)
        predictions = predictions[sorted_idx]
        durations = durations[sorted_idx]
        events = events[sorted_idx]
        
        exp_predictions = torch.exp(predictions)
        risk_cum = torch.cumsum(exp_predictions, dim=0)
        
        failure_idx = torch.nonzero(events == 1).squeeze()
        
        loss = -torch.sum(predictions[failure_idx] - torch.log(risk_cum[failure_idx]))
        return loss
    
class PCGrad():

    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def pc_backward(self, objectives):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                               for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                               for g in pc_grad]).sum(dim=0)
        else:
            exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad
    

import math

class SmoothStep(nn.Module):
    def __init__(self, gamma=1.0,device='cpu'):
        super(SmoothStep, self).__init__()
        self.gamma = gamma
        self.a = -2 / (gamma ** 3)
        self.b = 3 / (2 * gamma)
        self.c = 0.5
        self.device = device
    def forward(self, x):
        return torch.where(x <= -self.gamma/2, torch.zeros_like(x,device=self.device),
                           torch.where(x >= self.gamma/2, torch.ones_like(x,device=self.device),
                                       self.a * (x ** 3) + self.b * x + self.c))

class DSelectKGate(nn.Module):
    def __init__(self, num_experts, num_nonzeros, gamma=1.0, entropy_reg_weight=1e-6,device='cpu'):
        super(DSelectKGate, self).__init__()
        self.num_experts = num_experts
        self.num_nonzeros = num_nonzeros
        self.num_binary = math.ceil(math.log2(num_experts))
        self.is_power_of_2 = (num_experts == 2 ** self.num_binary)
        self.smooth_step = SmoothStep(gamma,device=device)
        self.z_logits = nn.Parameter(torch.randn(num_nonzeros, 1, self.num_binary) * gamma / 100)
        self.w_logits = nn.Parameter(torch.randn(num_nonzeros, 1))
        binary_matrix = torch.tensor([[int(b) for b in f'{i:0{self.num_binary}b}'] for i in range(num_experts)],device=device)
        self.binary_codes = binary_matrix.unsqueeze(0).bool()
        self.entropy_reg_weight = entropy_reg_weight
        self.to(device=device)

    def forward(self, experts):
        smooth_step_activations = self.smooth_step(self.z_logits)
        selector_outputs = torch.prod(torch.where(self.binary_codes, smooth_step_activations,
                                                  1 - smooth_step_activations), dim=2)
        selector_weights = torch.softmax(self.w_logits, dim=0)
        expert_weights = torch.sum(selector_weights * selector_outputs, dim=0)
        # print(self.num_experts,experts)
        output = sum(expert_weights[i] * experts[i] for i in range(self.num_experts))

        entropy_reg = torch.sum(selector_outputs * torch.log(selector_outputs + 1e-9))
        if not self.is_power_of_2:
            reachability_reg = torch.sum(1 / torch.sum(selector_outputs, dim=1))
        else:
            reachability_reg = 0

        return output, self.entropy_reg_weight * entropy_reg + reachability_reg, expert_weights
    
import numpy as np
import torch
import os

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss