import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import math
from hypernet import D_HyperNet, AdapterWithHyperNet
from our_model import MMoE, D_HyperNet, FeatureSelectionLayer, Tower, RevExpert, task_embedding, OmicsComboAttention, gateNet, RevBlock, MultiheadAttention


class LRPRule(ABC):

    @abstractmethod
    def backward(self, layer, relevance_output):
        pass

    def _check_input_relevance_shape(self, input_tensor, relevance_input):
        assert input_tensor.shape == relevance_input.shape, f"Input tensor shape {input_tensor.shape} does not match input relevance shape {relevance_input.shape}"

    def _check_output_relevance_shape(self, output_tensor, relevance_output):
        assert output_tensor.shape == relevance_output.shape, f"Output tensor shape {output_tensor.shape} does not match output relevance shape {relevance_output.shape}"

    def _check_relevance_conservation(self, relevance_input, relevance_output):
        relevance_input_sum = relevance_input.sum()
        relevance_output_sum = relevance_output.sum()
        assert torch.allclose(
            relevance_input_sum, relevance_output_sum, rtol=1e-3, atol=1e-5
        ), f"Relevance is not conserved: input relevance sum {relevance_input_sum}, output relevance sum {relevance_output_sum}"


class IdentityRule(LRPRule):

    def backward(self, layer, relevance):
        return relevance


class EpsilonRule(LRPRule):

    def __init__(self, epsilon=1e-7):
        self.epsilon = epsilon

    def backward(self, layer, relevance):
        input = layer.input[0]
        output = layer.output
        z = output + self.epsilon * torch.sign(output)
        s = relevance / z
        c = torch.matmul(s, layer.weight)

        relevance_input = c * input

        return relevance_input


class LinearRule(LRPRule):

    def __init__(self, epsilon=1e-7, with_bias=True):
        self.epsilon = epsilon
        self.with_bias = with_bias

    def backward(self, layer, relevance_output):
        input, output = layer.input[0], layer.output
        weight, bias = layer.weight, layer.bias

        eps = self.epsilon * torch.sign(output)

        if not self.with_bias:
            output = torch.matmul(input, weight.t()) + eps

        relevance_input = torch.einsum("ji, b...i, b...j -> b...i", weight,
                                       input,
                                       relevance_output / (output + eps))

        # normalization_factor = relevance_output.sum() / (relevance_input.sum() + self.epsilon)

        # relevance_input = relevance_input * normalization_factor

        return relevance_input


class LinearRule2(LRPRule):

    def __init__(self, epsilon=1e-7):
        self.epsilon = epsilon

    def backward(self, layer, relevance):
        input, output = layer.input[0], layer.output
        weight, bias = layer.weight, layer.bias
        z = torch.matmul(input, weight.t()) + self.epsilon
        s = relevance / z
        c = torch.matmul(s, weight)
        R = input * c
        return R


class GammaRule(LRPRule):

    def __init__(self, gamma=0.25):
        self.gamma = gamma

    def backward(self, layer, relevance):
        input = layer.input[0]
        output = layer.output
        z = layer.forward(input)

        if z > 0:
            r = relevance * (input * (layer.weight > 0) /
                             (z + self.gamma * (layer.weight > 0).sum()))
        else:
            r = relevance * (input * (layer.weight < 0) /
                             (z + self.gamma * (layer.weight < 0).sum()))

        return r.sum(dim=-1)


class AddRule(LRPRule):

    def backward(self, input_1, input_2, relevance, epsilon=1e-4):
        total_input = input_1 + input_2 + epsilon
        relevance_norm = relevance / total_input

        relevance_a = relevance_norm * input_1
        relevance_b = relevance_norm * input_2

        return relevance_a, relevance_b


class SoftmaxRule(LRPRule):

    def backward(self, layer, relevance):
        input = layer.input[0]
        output = layer.output
        r = relevance - output * relevance.sum(dim=-1, keepdim=True)
        r = r * input
        return r


class SigmoidRule(LRPRule):

    def backward(ctx, layer, out_relevance):
        input = layer.input[0]
        output = layer.output

        relevance = out_relevance * output * (1 - output)

        return relevance


class MatMulRule(LRPRule):

    def __init__(self, epsilon=1e-7):
        self.epsilon = epsilon

    def backward(self, layer, relevance):
        input1, input2 = layer.input
        output = layer.output
        z = output * 2 + self.epsilon
        relevance_norm = relevance / z
        r1 = torch.matmul(relevance_norm, input2.transpose(-1,
                                                           -2)).mul_(input1)
        r2 = torch.matmul(input1.transpose(-1, -2),
                          relevance_norm).mul_(input2)
        return r1, r2


class meanRule(LRPRule):

    def __init__(self, sum_dim=0, epsilon=1e-7):
        self.epsilon = epsilon
        self.dim = sum_dim

    def backward(self, inputs, relevance):
        r = []
        for input in inputs:
            input_weight = input / sum(inputs) + self.epsilon
            r.append(relevance * input_weight)
        return r


class NormRule(LRPRule):

    def backward(self, layer, relevance):
        return relevance


class ReluRule(LRPRule):

    def __init__(self, alpha=1, beta=1, epsilon=1e-7):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def backward(self, layer, relevance):
        input = layer.input[0]
        output = layer.output
        # z_pos = torch.clamp(input, min=0)
        # z_neg = torch.clamp(input, max=0)
        # relevance_pos = torch.matmul(
        #     relevance, self.alpha * z_pos /
        #     (z_pos.sum(dim=-1, keepdim=True) + self.epsilon)) * z_pos
        # relevance_neg = torch.matmul(
        #     relevance, self.beta * z_neg /
        #     (z_neg.sum(dim=-1, keepdim=True) - self.epsilon)) * z_neg
        # relevance = relevance_pos + relevance_neg
        # return relevance
        z_pos = torch.clamp(input, min=0)
        z_neg = torch.clamp(input, max=0)
        relevance_pos = (
            self.alpha * z_pos /
            (z_pos.sum(dim=-1, keepdim=True) + self.epsilon)) * relevance
        relevance_neg = (
            self.beta * z_neg /
            (z_neg.sum(dim=-1, keepdim=True) - self.epsilon)) * relevance
        relevance = relevance_pos + relevance_neg
        return relevance


class AttentionRule(LRPRule):

    def __init__(self, epsilon=1e-7):
        self.epsilon = epsilon

    def backward(self, layer, relevance_output):
        # out_proj
        relevance_output = LinearRule(self.epsilon).backward(
            layer.out_proj, relevance_output)
        relevance_output = relevance_output.view(relevance_output.size(0), -1,
                                                 layer.num_heads,
                                                 layer.head_dim).transpose(
                                                     1, 2).contiguous()

        # matmul (attn_probs and v)
        relevance_attn_scores, relevance_v = MatMulRule(self.epsilon).backward(
            layer.mm2, relevance_output)
        relevance_attn_scores = SoftmaxRule().backward(layer.softmax,
                                                       relevance_attn_scores)
        relevance_attn_scores *= math.sqrt(layer.head_dim)
        relevance_v = relevance_v.transpose(1, 2).contiguous().view(
            relevance_v.size(0), -1, layer.embed_dim)

        # matmul (q and k)
        relevance_q, relevance_k = MatMulRule(self.epsilon).backward(
            layer.mm1, relevance_attn_scores)
        relevance_q = relevance_q.transpose(1, 2).contiguous().view(
            relevance_q.size(0), -1, layer.embed_dim)
        relevance_k = relevance_k.transpose(-2, -1).transpose(
            1, 2).contiguous().view(relevance_k.size(0), -1, layer.embed_dim)

        # linear layers
        relevance_query = LinearRule(self.epsilon).backward(
            layer.q_proj, relevance_q)
        relevance_key = LinearRule(self.epsilon).backward(
            layer.k_proj, relevance_k)
        relevance_value = LinearRule(self.epsilon).backward(
            layer.v_proj, relevance_v)

        return relevance_query, relevance_key, relevance_value


class RevBlockRule(LRPRule):

    def backward(self, layer, relevance_y):
        relevance_y1, relevance_y2 = torch.chunk(relevance_y, 2, dim=1)

        relevance_x2 = relevance_y2 - layer.G(relevance_y1)
        relevance_x1 = relevance_y1 - layer.F(relevance_x2)

        relevance_x = torch.cat((relevance_x1, relevance_x2), dim=1)

        return relevance_x


class AdapterWithHyperNetRule(LRPRule):

    def __init__(self):
        super().__init__()
        self.add_rule = AddRule()
        self.sigmoid_rule = IdentityRule()

    def backward(self, layer, relevance, id):
        residual, output = layer.omics_res[id], layer.omics_out[id]
        relevance_residual, relevance_output = self.add_rule.backward(
            residual, output, relevance)

        output1 = torch.einsum('bij,bj->bi', layer.down_proj_matrix, residual)
        output2 = torch.einsum('bij,bj->bi', layer.up_proj_matrix, output1)

        relevance_in = torch.einsum('bij,bj,bi->bj', layer.up_proj_matrix,
                                    output1, relevance_output / output2)
        relevance_in = torch.einsum('bij,bj,bi->bj', layer.down_proj_matrix,
                                    residual, relevance_in / output1)

        return relevance_residual + relevance_in


class GatingNetworkRule(LRPRule):

    def backward(self, layer, relevance):
        input = layer.input[0]
        output = layer.output
        r_list = []
        for w in output.T:
            r_list.append(torch.einsum('ij,i->ij', relevance, w))
        return r_list


class OmicsComboAttentionRule(LRPRule):

    def __init__(self, epsilon=1e-7):
        self.epsilon = epsilon
        self.linear_rule = LinearRule(epsilon)
        self.attention_rule = AttentionRule(epsilon)
        self.mean_rule = meanRule(sum_dim=1, epsilon=epsilon)

    def backward(self, layer, relevance_output):
        relevance = self.linear_rule.backward(layer.output_layer,
                                              relevance_output)

        relevance_list = self.mean_rule.backward(layer.combo_outputs,
                                                 relevance)
        relevance = torch.stack(relevance_list, dim=1)

        relevance_query, relevance_key, relevance_value = self.attention_rule.backward(
            layer.combo_attention, relevance)

        relevance_combo_embs_q = self.linear_rule.backward(
            layer.combo_query, relevance_query)
        relevance_combo_embs_k = self.linear_rule.backward(
            layer.combo_key, relevance_key)
        relevance_combo_embs_v = self.linear_rule.backward(
            layer.combo_value, relevance_value)

        relevance_combo_embs = relevance_combo_embs_q + relevance_combo_embs_k + relevance_combo_embs_v

        num_combos = layer.num_omics * (layer.num_omics - 1) // 2
        relevance_omics = [
            torch.zeros_like(layer.omics[0]) for _ in range(layer.num_omics)
        ]

        combo_idx = 0
        for i in range(layer.num_omics):
            for j in range(i + 1, layer.num_omics):
                relevance_ij = relevance_combo_embs[:, combo_idx]
                relevance_i, relevance_j = torch.chunk(relevance_ij, 2, dim=-1)

                relevance_omics[i] += relevance_i
                relevance_omics[j] += relevance_j

                combo_idx += 1

        return relevance_omics

    def reverse_combo_embs(combo_embs, num_omics):
        batch_size, num_combos, combo_dim = combo_embs.shape
        input_dim = combo_dim // 2

        omics = torch.zeros(batch_size,
                            num_omics,
                            input_dim,
                            device=combo_embs.device)

        count = torch.zeros(num_omics, device=combo_embs.device)

        combo_idx = 0
        for i in range(num_omics):
            for j in range(i + 1, num_omics):
                omic_i, omic_j = torch.chunk(combo_embs[:, combo_idx],
                                             2,
                                             dim=-1)

                omics[:, i] += omic_i
                omics[:, j] += omic_j

                count[i] += 1
                count[j] += 1

                combo_idx += 1

        for i in range(num_omics):
            omics[:, i] /= count[i]

        return omics


class LRPForwardTracker:

    def __init__(self, model):
        self.layer_order = []
        self.module_list = self.init_module_list(model)
        self.hooks = []

    def init_module_list(self, model):
        module_list = []
        for layer in model.children():
            if isinstance(layer, nn.ModuleList):
                for m in layer:
                    if isinstance(m, nn.ModuleList):
                        for mm in m:
                            module_list.append(mm)
                    else:
                        module_list.append(m)
            elif isinstance(layer, (nn.ParameterList, D_HyperNet)):
                pass
            else:
                module_list.append(layer)
        return module_list

    def register_hook(self, module):

        def hook(module, input, output):
            if module not in self.layer_order:
                self.layer_order.append(module)

        return hook

    def register_task_module_hook(self, task_id):
        for module in self.module_list:
            if isinstance(
                    module,
                (task_embedding, OmicsComboAttention, gateNet, Tower)):
                if module.task_id == task_id or module.task_id == None:
                    handle = module.register_forward_hook(
                        self.register_hook(module))
                    self.hooks.append(handle)
            else:
                handle = module.register_forward_hook(
                    self.register_hook(module))
                self.hooks.append(handle)

    def unregister_all_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()


class LRP:

    def __init__(self, model):
        self.model = model
        self.model_name = []
        self.rules = {
            nn.Linear: LinearRule(),
            nn.ReLU: ReluRule(),
            nn.Softmax: SoftmaxRule(),
            nn.Dropout: IdentityRule(),
            nn.BatchNorm1d: NormRule(),
            OmicsComboAttention: OmicsComboAttentionRule(),
            RevBlock: RevBlockRule(),
            AdapterWithHyperNet: AdapterWithHyperNetRule(),
            gateNet: GatingNetworkRule(),
        }

    def explain(self, input, task_id):

        def apply_rule(layer, relevance):
            if type(layer) in self.rules:
                relevance = self.rules[type(layer)].backward(layer, relevance)
            elif type(layer) in [Tower, FeatureSelectionLayer, task_embedding]:
                for module in reversed(list(layer.children())):
                    # print(module.__class__.__name__)

                    relevance = apply_rule(module, relevance)
            elif type(layer) in [RevExpert]:
                for module in reversed(layer.blocks):
                    # print(module.__class__.__name__)
                    relevance = apply_rule(module, relevance)
            else:
                raise NotImplementedError(
                    f"Unknown layer type: {layer.__class__.__name__}")
            return relevance

        hooks = self.register_hooks()
        tracker = LRPForwardTracker(self.model)
        tracker.unregister_all_hooks()
        tracker.register_task_module_hook(task_id)

        pred = self.model(*input)

        relevance_out = pred

        relevance = relevance_out[0][task_id]
        weighted_expert_outs = None
        weighted_expert_outs_sum = None
        experts = []
        expert_relevances = []
        embedding_list = []
        attn_list = []

        for module in reversed(tracker.layer_order):
            # print(module.__class__.__name__)

            if type(module) == gateNet:
                weighted_expert_outs_sum = module.output
                weighted_expert_outs = torch.unbind(
                    module.weighted_expert_outs, dim=1)

            elif type(module) == RevExpert:
                experts.append(module)

            elif type(module) == task_embedding:
                assert len(weighted_expert_outs) == len(
                    experts
                ), f"weighted_expert_outs:{len(weighted_expert_outs)},experts:{len(experts)}"
                for out in weighted_expert_outs:
                    expert_relevances.append(relevance * out /
                                             weighted_expert_outs_sum)
                for i, (expert_relevance, expert) in enumerate(
                        zip(expert_relevances, reversed(experts))):
                    expert_relevances[i] = apply_rule(expert, expert_relevance)
                specific_relevance = sum(
                    expert_relevances[:self.model.num_specific_experts])
                shared_relevance = sum(
                    expert_relevances[self.model.num_specific_experts:])
                embedding_list.append(module)

            elif type(module) == OmicsComboAttention:
                attn_list.append(module)

            elif type(module) == AdapterWithHyperNet:
                specific_emb_relevance, specific_attn_relevance = AddRule(
                ).backward(embedding_list[1].output, attn_list[1].output,
                           specific_relevance)
                shared_emb_relevance, shared_attn_relevance = AddRule(
                ).backward(embedding_list[0].output, attn_list[0].output,
                           shared_relevance)
                specific_emb_relevance = apply_rule(embedding_list[1],
                                                    specific_emb_relevance)
                specific_attn_relevances = apply_rule(attn_list[1],
                                                      specific_attn_relevance)
                shared_emb_relevance = apply_rule(embedding_list[0],
                                                  shared_emb_relevance)
                shared_attn_relevances = apply_rule(attn_list[0],
                                                    shared_attn_relevance)
                emb_relevance = specific_emb_relevance + shared_emb_relevance
                attn_relevances = [
                    r_sp + r_sh for r_sp, r_sh in zip(specific_attn_relevances,
                                                      shared_attn_relevances)
                ]
                emb_relevances = list(torch.chunk(emb_relevance, 4, dim=1))
                for i in range(len(attn_relevances)):
                    attn_relevances[i] = AdapterWithHyperNetRule().backward(
                        module, attn_relevances[i], str(i))
                for i in range(len(emb_relevances)):
                    emb_relevances[i] = AdapterWithHyperNetRule().backward(
                        module, emb_relevances[i], str(i))
                relevances = [
                    attn + emb
                    for attn, emb in zip(attn_relevances, emb_relevances)
                ]
                index = len(relevances) - 1
            elif type(module) == FeatureSelectionLayer:
                relevances[index] = apply_rule(module, relevances[index])
                index -= 1
            else:
                relevance = apply_rule(module, relevance)

        return relevances

    def register_hooks(self):
        hooks = []

        def generic_hook(module, input, output):
            module.input = input
            module.output = output

        def name_hook(module, input, output):
            self.model_name.append(module.__class__.__name__)

        def register_hook(module):
            if isinstance(module,
                          (nn.Linear, nn.LayerNorm, nn.BatchNorm1d, nn.ReLU,
                           nn.Softmax, nn.Sigmoid, nn.Dropout,
                           AdapterWithHyperNet, RevBlock, MultiheadAttention)):
                hook_handle = module.register_forward_hook(generic_hook)
                # module.register_backward_hook(backward_hook)
                hooks.append(hook_handle)
            elif isinstance(module, ()):
                pass
            elif isinstance(module, (nn.ModuleList, nn.Sequential)):
                for sub_module in module:
                    register_hook(sub_module)
            else:
                hook_handle = module.register_forward_hook(generic_hook)
                hooks.append(hook_handle)
                for child in module.children():
                    register_hook(child)

        self.model.apply(register_hook)

        return hooks
