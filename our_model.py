import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from data_utils import get_cancer_datasets
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from lifelines.utils import concordance_index
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, roc_auc_score
from utils import set_seed, PCGrad, PriorMultiLabelSoftMarginLoss, ImprovedDeepSurvLoss, EarlyStopping
from typing import Any, Tuple
from hypernet import D_HyperNet, AdapterWithHyperNet
import json
import math

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detach_and_grad(inputs: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = True
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ",
            type(inputs).__name__)


class RevBlockFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, F, G, *params):
        with torch.no_grad():
            x1, x2 = torch.chunk(x, 2, dim=1)
            y1 = x1 + F(x2)
            y2 = x2 + G(y1)
            y = torch.cat([y1, y2], dim=1)

            x1.set_()
            x2.set_()
            y1.set_()
            y2.set_()
            del x1, x2, y1, y2

        ctx.save_for_backward(x, y)
        ctx.F = F
        ctx.G = G

        return y

    @staticmethod
    def backward(ctx, grad_y):
        F = ctx.F
        G = ctx.G
        x, y = ctx.saved_tensors
        x, y = detach_and_grad((x, y))

        y1, y2 = torch.chunk(y, 2, dim=1)
        with torch.no_grad():
            x2 = y2 - G(y1)
            x1 = y1 - F(x2)

        with torch.enable_grad():
            x1.requires_grad = True
            x2.requires_grad = True
            y1_ = x1 + F(x2)
            y2_ = x2 + G(y1)
            y = torch.cat((y1_, y2_), dim=1)
            grad = torch.autograd.grad(y, (x1, x2) + tuple(F.parameters()) +
                                       tuple(G.parameters()), grad_y)
            params_len = len(list(F.parameters()))
            grad_x1, grad_x2 = grad[:2]
            grad_x = torch.cat((grad_x1, grad_x2), dim=1)
            grad_f_params = grad[2:2 + params_len]
            grad_g_params = grad[2 + params_len:]

            y1_.detach_()
            y2_.detach_()
            del y1_, y2_

        x.data.set_(torch.cat((x1, x2), dim=1).data.contiguous())
        return grad_x, None, None, *grad_f_params, *grad_g_params


class RevBlock(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, Dropout=0):
        super().__init__()
        self.F = nn.Sequential(nn.Linear(in_dim // 2, hidden_dim),
                               nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                               nn.Dropout(Dropout),
                               nn.Linear(hidden_dim, in_dim // 2),
                               nn.BatchNorm1d(in_dim // 2), nn.ReLU())
        self.G = nn.Sequential(nn.Linear(in_dim // 2, hidden_dim),
                               nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                               nn.Dropout(Dropout),
                               nn.Linear(hidden_dim, in_dim // 2),
                               nn.BatchNorm1d(in_dim // 2), nn.ReLU())

    def forward(self, x):
        params = list(self.F.parameters()) + list(self.G.parameters())
        return RevBlockFunction.apply(x, self.F, self.G, *params)


class gateNet(nn.Module):

    def __init__(self,
                 in_dim,
                 hidden_dim,
                 num_experts,
                 task_id,
                 task_feat_nums: list,
                 num_views=2):  #in_dim = x_dim + experts_out_dim*num_experts
        super(gateNet, self).__init__()
        self.num_views = num_views
        self.task_id = task_id
        self.task_embeddings = nn.ModuleList(
            [nn.Embedding(num, hidden_dim) for num in task_feat_nums])

        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm1d(in_dim + hidden_dim * len(task_feat_nums)),
                nn.ReLU(),
                nn.Linear(in_dim + hidden_dim * len(task_feat_nums),
                          hidden_dim), nn.ReLU()) for i in range(num_views)
        ])

        self.gate_out = nn.Sequential(
            nn.Linear(hidden_dim * num_views, num_experts), nn.Sigmoid())

    def forward(self, x, task_feats_list: list, expert_outs_detach: list,
                expert_outs: list):
        task_embs = []
        for feat, emb in zip(task_feats_list, self.task_embeddings):
            task_emb = emb(feat)
            task_embs.append(task_emb)
        task_embs = torch.cat(task_embs, dim=1)
        x = torch.cat([x.detach(), task_embs], dim=1)
        expert_x = torch.cat(expert_outs_detach, dim=1)
        x = torch.cat([x, expert_x], dim=1)

        x_list = []
        for i in range(self.num_views):
            x_list.append(self.fc[i](x))

        x = torch.cat(x_list, dim=1)
        g_w = self.gate_out(x).unsqueeze(-1)
        expert_outs = torch.stack(expert_outs, dim=1)
        weighted_expert_outs = g_w * expert_outs
        self.weighted_expert_outs = weighted_expert_outs
        output = torch.sum(weighted_expert_outs, dim=1)

        return output


class RevExpert(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_blocks, task_id, Dropout=0):
        super().__init__()
        self.blocks = nn.ModuleList([
            RevBlock(in_dim, hidden_dim, in_dim, Dropout)
            for _ in range(num_blocks)
        ])
        self.task_id = task_id

    def forward(self, x):
        outs = []
        for block in self.blocks:
            x = block(x.clone())
            outs.append(x.clone())
        return x


class Tower(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 task_id,
                 Dropout=0):
        super(Tower, self).__init__()
        self.task_id = task_id
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()  #nn.ELU()
        self.Dropout = nn.Dropout(Dropout)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.Dropout(x)
        x = self.linear2(x)
        return x


class matmul(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.matmul(a, b)


def save_io_hook(module, input, output):
    module.input = input
    module.output = output


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.mm1 = matmul()
        self.mm2 = matmul()
        self.softmax = nn.Softmax(dim=-1)

        # 为每个子模块注册钩子
        self.q_proj.register_forward_hook(save_io_hook)
        self.k_proj.register_forward_hook(save_io_hook)
        self.v_proj.register_forward_hook(save_io_hook)
        self.out_proj.register_forward_hook(save_io_hook)
        self.softmax.register_forward_hook(save_io_hook)
        self.mm1.register_forward_hook(save_io_hook)
        self.mm2.register_forward_hook(save_io_hook)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(batch_size, -1, self.num_heads,
                   self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads,
                   self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads,
                   self.head_dim).transpose(1, 2)

        attn_scores = self.mm1(q, k.transpose(-2, -1)) / math.sqrt(
            self.head_dim)
        attn_scores = self.softmax(attn_scores)
        attn_output = self.mm2(attn_scores, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output


class OmicsComboAttention(nn.Module):

    def __init__(self, num_omics, input_dim, out_dim, num_heads, task_id):
        super().__init__()
        self.num_omics = num_omics
        self.task_id = task_id
        self.combo_query = nn.Linear(input_dim * 2, input_dim)
        self.combo_key = nn.Linear(input_dim * 2, input_dim)
        self.combo_value = nn.Linear(input_dim * 2, input_dim)
        self.combo_attention = MultiheadAttention(input_dim, num_heads)
        self.output_layer = nn.Linear(input_dim, out_dim)

    def forward(self, omics: list):
        self.omics = omics
        omics = torch.stack(omics, dim=1)  # (batch_size, num_omics, input_dim)
        combo_embs = []
        for i in range(self.num_omics):
            for j in range(i + 1, self.num_omics):
                combo_emb = torch.cat([omics[:, i], omics[:, j]], dim=-1)
                combo_embs.append(combo_emb)
        combo_embs = torch.stack(
            combo_embs, dim=1)  # (batch_size, num_combos, input_dim * 2)

        query = self.combo_query(combo_embs)
        key = self.combo_key(combo_embs)
        value = self.combo_value(combo_embs)
        combo_output = self.combo_attention(query, key, value)
        self.combo_outputs = torch.unbind(combo_output, dim=1)
        combo_output = combo_output.mean(dim=1)  # (batch_size, input_dim)

        # 经过输出层
        output = self.output_layer(combo_output)
        return output


class FeatureSelectionLayer(nn.Module):

    def __init__(self, input_dim, output_dim, p=1, lambda_=1e-4):
        super(FeatureSelectionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.p = p
        self.lambda_ = lambda_
        self.epsilon = 1e-8

        self.fc = nn.Linear(input_dim, output_dim)
        self.rulu = nn.ReLU()

    def forward(self, x):
        x = self.rulu(self.fc(x))
        return x

    def feature_importance(self):
        weights = self.fc.weight.data
        feature_importance = torch.norm(weights, p=2, dim=1)
        return feature_importance

    def regularization_loss(self):
        weights = self.fc.weight
        norm = torch.sum(
            torch.pow(torch.sum(weights**2, dim=1) + self.epsilon, self.p / 2))
        return self.lambda_ * norm


class task_embedding(nn.Module):

    def __init__(self, in_dim, emb_dim, task_id):
        super().__init__()
        self.task_id = task_id
        self.linear = nn.Linear(in_dim, emb_dim)

    def forward(self, x):
        x = self.linear(x)
        return x


# MMoE模型
class MMoE(nn.Module):

    def __init__(self,
                 in_dims,
                 bottleneck_dim,
                 num_shared_experts,
                 num_specific_experts,
                 emb_dim,
                 task_output_sizes,
                 num_blocks,
                 hidden_dim,
                 task_feat_nums: list,
                 Dropout=0):
        super().__init__()
        self.num_shared_experts = num_shared_experts
        self.num_specific_experts = num_specific_experts
        self.num_tasks = len(task_output_sizes)
        self.num_experts = num_shared_experts + num_specific_experts * self.num_tasks
        self.emb_dim = emb_dim
        self.fc_in = nn.ModuleList([
            FeatureSelectionLayer(in_dim, bottleneck_dim, p=1)
            for in_dim in in_dims
        ])
        self.task_embeddings = nn.ModuleList([
            task_embedding(bottleneck_dim * 4, emb_dim, i) for i in [
                i if i < self.num_tasks else None
                for i in range(self.num_tasks + 1)
            ]
        ])
        self.task_attentions = nn.ModuleList([
            OmicsComboAttention(4, bottleneck_dim, emb_dim, 4, i) for i in [
                i if i < self.num_tasks else None
                for i in range(self.num_tasks + 1)
            ]
        ])
        self.shared_experts = nn.ModuleList([
            RevExpert(emb_dim, hidden_dim, num_blocks, None, Dropout)
            for _ in range(num_shared_experts)
        ])
        self.specific_experts = nn.ModuleList([
            nn.ModuleList([
                RevExpert(emb_dim, hidden_dim, num_blocks, i, Dropout)
                for _ in range(num_specific_experts)
            ]) for i in range(self.num_tasks)
        ])
        self.gates = nn.ModuleList([
            gateNet(emb_dim * (self.num_tasks + 1) +
                    self.num_experts * emb_dim,
                    hidden_dim,
                    self.num_experts,
                    i,
                    task_feat_nums,
                    num_views=2) for i in range(self.num_tasks)
        ])
        self.task_layers = nn.ModuleList([
            Tower(emb_dim, emb_dim // 2, out_dim, task_id, Dropout)
            for task_id, out_dim in enumerate(task_output_sizes)
        ])
        self.hypernet = D_HyperNet(1, [64, 32, 64], 8)
        self.adapter = AdapterWithHyperNet(bottleneck_dim, 64, 8)
        self.log_vars = nn.ParameterList([
            nn.Parameter(torch.zeros(1, requires_grad=True))
            for _ in task_output_sizes
        ])
        self.to(device=device)

    def forward(self, inputs, domain_feat, task_feats):
        x_list = []
        fc_reg_losses = []
        for x, fc in zip(inputs, self.fc_in):
            x = fc(x)
            x_list.append(x)
            fc_reg_losses.append(fc.regularization_loss())

        adapter_param = self.hypernet(domain_feat)
        x_list = [
            self.adapter(x, adapter_param, str(id))
            for id, x in enumerate(x_list)
        ]

        task_att_list = []
        for task_att in self.task_attentions:
            task_att_list.append(task_att(x_list))

        x = torch.cat(x_list, dim=1)
        task_emb_list = []
        for task_emb in self.task_embeddings:
            task_emb_list.append(task_emb(x))

        task_emb_list = [
            task_emb_list[i] + task_att_list[i]
            for i in range(self.num_tasks + 1)
        ]
        specific_expert_outputs, shared_expert_outputs = [], []
        for i in range(self.num_tasks):
            task_expert_outputs = []
            for j in range(self.num_specific_experts):
                task_expert_outputs.append(self.specific_experts[i][j](
                    task_emb_list[i]))
            specific_expert_outputs.append(task_expert_outputs)
        for i in range(self.num_shared_experts):
            shared_expert_outputs.append(self.shared_experts[i](
                task_emb_list[-1]))

        for i in range(self.num_tasks):
            gate_input_detach = []
            for j in range(self.num_tasks):
                if j == i:
                    gate_input_detach.extend(specific_expert_outputs[j])
                else:
                    specific_expert_outputs_j = specific_expert_outputs[j]
                    specific_expert_outputs_j = [
                        out.detach() for out in specific_expert_outputs_j
                    ]
                    gate_input_detach.extend(specific_expert_outputs_j)
            gate_input_detach.extend(shared_expert_outputs)

        gate_input = []
        for j in range(self.num_tasks):
            gate_input.extend(specific_expert_outputs[j])
        gate_input.extend(shared_expert_outputs)

        task_outputs = []
        for task_id, (gate, task_network, task_feat) in enumerate(
                zip(self.gates, self.task_layers, task_feats)):
            task_id = torch.tensor(task_id,
                                   device=device).repeat(task_feat.size(0))
            x_c = torch.cat(task_emb_list, dim=1)
            gated_output = gate(x_c, [task_id, task_feat], gate_input_detach,
                                gate_input)
            task_output = task_network(gated_output)
            task_outputs.append(task_output)
        return task_outputs, sum(fc_reg_losses)

    def shared_parameters(self):
        shared_params = []

        for fc in self.fc_in:
            shared_params.extend(fc.parameters())

        shared_params.extend(self.hypernet.parameters())

        shared_params.extend(self.task_embeddings[-1].parameters())
        shared_params.extend(self.task_attentions[-1].parameters())

        for expert in self.shared_experts:
            shared_params.extend(expert.parameters())

        return shared_params

    def task_specific_parameters(self, task_id=None):
        if task_id is None:
            task_specific_params = []
            for i in range(self.num_tasks):
                task_params = []
                # specific_experts
                for expert in self.specific_experts[i]:
                    task_params.extend(expert.parameters())

                # task_embedding
                task_params.extend(self.task_embeddings[i].parameters())
                # task_attention
                task_params.extend(self.task_attentions[i].parameters())

                # gates
                task_params.extend(self.gates[i].parameters())

                # task_layers
                task_params.extend(self.task_layers[i].parameters())

                task_specific_params.extend(task_params)

            return task_specific_params

        else:
            task_params = []

            # specific_experts
            for expert in self.specific_experts[task_id]:
                task_params.extend(expert.parameters())

            # task_embedding
            task_params.extend(self.task_embeddings[task_id].parameters())
            # task_attention
            task_params.extend(self.task_attentions[task_id].parameters())

            # gates
            task_params.extend(self.gates[task_id].parameters())

            # task_layers
            task_params.extend(self.task_layers[task_id].parameters())

            return task_params

    def evaluate(self, model_outputs, all_labels, task_types):

        metrics = {}

        # 计算每个任务的评估指标
        for i, task_type in enumerate(task_types):
            model_outputs[i] = model_outputs[i].cpu().detach().numpy()
            all_labels[i] = all_labels[i].cpu().detach().numpy()
            if task_type == 'binary':
                pred_labels = np.argmax(model_outputs[i], axis=1)
                true_labels = all_labels[i]
                accuracy = accuracy_score(true_labels, pred_labels)
                f1 = f1_score(true_labels, pred_labels, average='macro')

                metrics[f"task_{i+1}"] = {"accuracy": accuracy, "f1": f1}
                print(f"Task {i+1} ({task_type}):")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"F1 Score: {f1:.4f}")
            elif task_type in ['multiclass', "weighted_multiclass"]:
                pred_labels = np.argmax(model_outputs[i], axis=1)
                true_labels = all_labels[i]
                probas = softmax(model_outputs[i], axis=1)
                accuracy = accuracy_score(true_labels, pred_labels)
                f1 = f1_score(true_labels, pred_labels, average='weighted')
                auc = roc_auc_score(true_labels,
                                    probas,
                                    multi_class='ovr',
                                    average='weighted')
                precision = precision_score(true_labels,
                                            pred_labels,
                                            average='weighted',
                                            zero_division=1)
                recall = recall_score(true_labels,
                                      pred_labels,
                                      average='weighted',
                                      zero_division=1)

                metrics[f"task_{i+1}"] = {
                    "accuracy": accuracy,
                    "f1": f1,
                    "auc": auc,
                    "precision": precision,
                    "recall": recall
                }
                print(f"Task {i+1} ({task_type}):")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"F1 Score: {f1:.4f}")
                print(f"AUC: {auc:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
            elif task_type == 'survival':
                # Concordance Index
                test_time = all_labels[i][:, 0]
                test_event = all_labels[i][:, 1]
                ci = concordance_index(test_time, -model_outputs[i],
                                       test_event)

                metrics[f"task_{i+1}"] = {"c_index": ci}
                print(f"Task {i+1} ({task_type}):")
                print(f"C-index: {ci:.4f}")

            elif task_type == "regression":
                mse = mean_squared_error(all_labels[i], model_outputs[i])
                mae = mean_absolute_error(all_labels[i], model_outputs[i])
                r2 = r2_score(all_labels[i], model_outputs[i])
                metrics[f"task_{i+1}"] = {"mse": mse, "mae": mae, "r2": r2}
                print(f"Task {i+1} ({task_type}):")
                print(f"Mean Squared Error: {mse:.4f}")
                print(f"Mean Absolute Error: {mae:.4f}")
                print(f"R2 Score: {r2:.4f}")
            else:
                raise (NotImplementedError(
                    f"Task name {task_type} not implemented."))
            # print()
        return metrics


def uncertainty_weighted_loss(loss, log_var):
    return torch.exp(-log_var) * loss**2 + log_var


def train_epoch(model, train_loader, optimizer, criterion_list, device,
                task_type_map, task_types):
    model.train()
    losses = [0] * len(criterion_list)
    reg_loss_total = 0
    for task_inputs1, task_inputs2, task_inputs3, task_inputs4, label in train_loader:
        task_inputs = [
            task_inputs1.to(device),
            task_inputs2.to(device),
            task_inputs3.to(device),
            task_inputs4.to(device)
        ]
        domain_feat = label[-1].unsqueeze(1).to(device)
        task_feats = [
            torch.tensor(task_type_map[i],
                         device=device).repeat(task_inputs1.size(0))
            for i in task_types
        ]

        y_pred, reg_loss = model(task_inputs, domain_feat, task_feats)

        label_list = [[label[-4].to(device)],
                      [
                          label[-3][:, 0].unsqueeze(1).to(device),
                          label[-3][:, 1].unsqueeze(1).to(device)
                      ], [label[-2].unsqueeze(1).to(device)]]

        task_losses = [
            criterion(y_pred[i], *label)
            for i, (criterion,
                    label) in enumerate(zip(criterion_list, label_list))
        ]
        loss = task_losses + [reg_loss]

        optimizer.zero_grad()
        optimizer.pc_backward(loss)
        optimizer.step()

        for i, l in enumerate(task_losses):
            losses[i] += l.item()
        reg_loss_total += reg_loss.item()

    avg_losses = [l / len(train_loader) for l in losses]
    avg_reg_loss = reg_loss_total / len(train_loader)
    return avg_losses, avg_reg_loss


def validate_epoch(model, val_loader, criterion_list, device, task_type_map,
                   task_types):
    model.eval()
    losses = [0] * len(criterion_list)
    reg_loss_total = 0
    with torch.no_grad():
        for task_inputs1, task_inputs2, task_inputs3, task_inputs4, label in val_loader:
            task_inputs = [
                task_inputs1.to(device),
                task_inputs2.to(device),
                task_inputs3.to(device),
                task_inputs4.to(device)
            ]
            domain_feat = label[-1].unsqueeze(1).to(device)
            task_feats = [
                torch.tensor(task_type_map[i],
                             device=device).repeat(task_inputs1.size(0))
                for i in task_types
            ]

            y_pred, reg_loss = model(task_inputs, domain_feat, task_feats)

            label_list = [[label[-4].to(device)],
                          [
                              label[-3][:, 0].unsqueeze(1).to(device),
                              label[-3][:, 1].unsqueeze(1).to(device)
                          ], [label[-2].unsqueeze(1).to(device)]]

            task_losses = [
                criterion(y_pred[i], *label)
                for i, (criterion,
                        label) in enumerate(zip(criterion_list, label_list))
            ]

            for i, l in enumerate(task_losses):
                losses[i] += l.item()
            reg_loss_total += reg_loss.item()

    avg_losses = [l / len(val_loader) for l in losses]
    avg_reg_loss = reg_loss_total / len(val_loader)
    return avg_losses, avg_reg_loss


def train_model(model, train_loader, val_loader, optimizer, criterion_list,
                num_epochs, task_weights, device, fold, task_type_map,
                task_types):
    early_stopping = EarlyStopping(patience=30,
                                   verbose=True,
                                   path=f'./saved/best_model_fold_{fold}.pt')

    for epoch in range(num_epochs):
        train_losses, train_reg_loss = train_epoch(model, train_loader,
                                                   optimizer, criterion_list,
                                                   device, task_type_map,
                                                   task_types)
        val_losses, val_reg_loss = validate_epoch(model, val_loader,
                                                  criterion_list, device,
                                                  task_type_map, task_types)

        weighted_val_loss = sum(
            loss * weight
            for loss, weight in zip(val_losses, task_weights)) + val_reg_loss

        print(f'Fold {fold+1}, Epoch {epoch+1}/{num_epochs}')
        print(f'Train Losses: {train_losses}, Reg Loss: {train_reg_loss}')
        print(f'Val Losses: {val_losses}, Reg Loss: {val_reg_loss}')
        print(f'Weighted Val Loss: {weighted_val_loss}')

        early_stopping(weighted_val_loss, model)

        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load(f'./saved/best_model_fold_{fold}.pt'))
    return model


def convert_to_json_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {
            key: convert_to_json_serializable(value)
            for key, value in obj.items()
        }
    else:
        return obj


def train_and_evaluate(data_dict, config):
    # 解析配置
    in_dim = config['in_dim']
    bottleneck_dim = config['bottleneck_dim']
    emb_dim = config['emb_dim']
    task_output_sizes = config['task_output_sizes']
    num_shared_experts = config['num_shared_experts']
    num_specific_experts = config['num_specific_experts']
    num_blocks = config['num_blocks']
    hidden_dim = config['hidden_dim']
    Dropout = config['Dropout']
    task_feat_nums = config['task_feat_nums']
    criterion_list = config['criterion_list']
    lr = config['lr']
    n_splits = config['n_splits']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    task_types = config['task_types']
    task_type_map = config['task_type_map']
    task_weights = config['task_weights']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = data_dict["dataset"]

    data_type = data.labels[1].cpu().numpy()

    kf = StratifiedShuffleSplit(n_splits=n_splits,
                                test_size=0.2,
                                random_state=42)

    fold_metrics = {}

    all_fold_metrics = {}

    for fold, (train_indices,
               val_indices) in enumerate(kf.split(data, data_type)):
        print(f"Fold {fold+1}/{n_splits}")

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(data,
                                  batch_size=batch_size,
                                  sampler=train_sampler)
        val_loader = DataLoader(data,
                                batch_size=len(val_indices),
                                sampler=val_sampler)

        model = MMoE(in_dim,
                     bottleneck_dim,
                     emb_dim=emb_dim,
                     num_shared_experts=num_shared_experts,
                     num_specific_experts=num_specific_experts,
                     task_output_sizes=task_output_sizes,
                     num_blocks=num_blocks,
                     hidden_dim=hidden_dim,
                     task_feat_nums=task_feat_nums,
                     Dropout=Dropout).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        pc_optimizer = PCGrad(optimizer)
        model = train_model(model, train_loader, val_loader, pc_optimizer,
                            criterion_list, num_epochs, task_weights, device,
                            fold, task_type_map, task_types)

        model.eval()
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for task_inputs1, task_inputs2, task_inputs3, task_inputs4, label in val_loader:
                task_inputs = [
                    task_inputs1.to(device),
                    task_inputs2.to(device),
                    task_inputs3.to(device),
                    task_inputs4.to(device)
                ]
                domain_feat = label[-1].unsqueeze(1).to(device)
                task_feats = [
                    torch.tensor(task_type_map[i],
                                 device=device).repeat(task_inputs1.size(0))
                    for i in task_types
                ]

                outputs, _ = model(task_inputs, domain_feat, task_feats)

                all_outputs.append(outputs)
                all_labels.append([label[-4], label[-3], label[-2]])

        all_outputs = [
            torch.cat([batch[i] for batch in all_outputs], dim=0)
            for i in range(len(all_outputs[0]))
        ]
        all_labels = [
            torch.cat([batch[i] for batch in all_labels], dim=0)
            for i in range(len(all_labels[0]))
        ]

        metrics = model.evaluate(all_outputs, all_labels, task_types)

        fold_metrics = convert_to_json_serializable(metrics)

        with open(f"./result/metrics_fold_{fold}.json", 'w',
                  encoding='utf-8') as json_file:
            json.dump(fold_metrics, json_file, ensure_ascii=False, indent=4)

        all_fold_metrics[f"fold_{fold}"] = fold_metrics

    avg_metrics = {}
    if all_fold_metrics:
        first_fold_metrics = all_fold_metrics["fold_0"]
        for task in first_fold_metrics.keys():
            avg_metrics[task] = {}
            for metric in first_fold_metrics[task].keys():
                avg_metrics[task][metric] = np.mean([
                    all_fold_metrics[f"fold_{i}"][task][metric]
                    for i in range(n_splits)
                ])

    avg_metrics = convert_to_json_serializable(avg_metrics)

    with open("./result/avg_metrics.json", 'w', encoding='utf-8') as json_file:
        json.dump(avg_metrics, json_file, ensure_ascii=False, indent=4)

    with open("./result/all_fold_metrics.json", 'w',
              encoding='utf-8') as json_file:
        json.dump(all_fold_metrics, json_file, ensure_ascii=False, indent=4)

    return all_fold_metrics, avg_metrics


if __name__ == "__main__":
    data_dict = get_cancer_datasets()

    config = {
        'in_dim': [19199, 335886, 640, 24776],
        'bottleneck_dim':
        64,
        'emb_dim':
        32,
        'task_output_sizes': [3, 1, 1],
        'num_shared_experts':
        4,
        'num_specific_experts':
        2,
        'num_blocks':
        8,
        'hidden_dim':
        8,
        'Dropout':
        0,
        'task_feat_nums': [3, 3],
        'criterion_list': [
            PriorMultiLabelSoftMarginLoss(
                prior=[300 / 599, 234 / 599, 65 / 599]),
            ImprovedDeepSurvLoss(),
            nn.L1Loss()
        ],
        'lr':
        1e-4,
        'n_splits':
        5,
        'batch_size':
        1024,
        'num_epochs':
        200,
        'task_types': ["multiclass", "survival", "regression"],
        'task_type_map': {
            "multiclass": 0,
            "survival": 1,
            "regression": 2
        },
        'task_weights': [0.5, 0.45, 0.05]
    }

    results = train_and_evaluate(data_dict, config)

    print("Final Results:")
    print(json.dumps(results, indent=4))
