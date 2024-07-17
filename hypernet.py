import torch
import torch.nn as nn

class D_HyperNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, k, dropout=0):
        super(D_HyperNet, self).__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, dim))
            # self.layers.append(nn.BatchNorm1d(dim))
            self.layers.append(nn.ELU())
            self.layers.append(nn.Dropout(dropout))
            prev_dim = dim
        self.layers.append(nn.Linear(prev_dim, k * k))
        self.k = k

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(-1, self.k, self.k)
        return x
    

class AdapterWithHyperNet(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, k):
        super(AdapterWithHyperNet, self).__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.k = k

        self.down_proj_left = nn.Parameter(torch.ones((bottleneck_dim, k)), requires_grad=True)
        self.down_proj_right = nn.Parameter(torch.ones((k, input_dim)), requires_grad=True)
        self.up_proj_left = nn.Parameter(torch.ones((input_dim, k)), requires_grad=True)
        self.up_proj_right = nn.Parameter(torch.ones((k, bottleneck_dim)), requires_grad=True)

        self.bias1 = nn.Parameter(torch.zeros(bottleneck_dim), requires_grad=True)
        self.bias2 = nn.Parameter(torch.zeros(input_dim), requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(input_dim), requires_grad=True)
        self.eps = 1e-5
        self.omics_res = {}
        self.omics_out = {}

    def forward(self, x, hypernet_output, omics_id):
        # 记录输入x,用于最后的residual connection
        self.residual = x
        self.omics_res[omics_id] = self.residual

        # Adapter layer-1: Down projection
        down_proj_matrix = torch.einsum('ik,bkj,jl->bil', self.down_proj_left, hypernet_output, self.down_proj_right)
        self.down_proj_matrix = down_proj_matrix

        z = torch.einsum('bij,bj->bi', down_proj_matrix, x)
        z = z + self.bias1

        # Adapter layer-2: Non-linear activation
        z = torch.sigmoid(z)

        # Adapter layer-3: Up projection
        up_proj_matrix = torch.einsum('ik,bkj,jl->bil', self.up_proj_left, hypernet_output, self.up_proj_right)
        self.up_proj_matrix = up_proj_matrix

        output = torch.einsum('bij,bj->bi', up_proj_matrix, z)
        output = output + self.bias2

        # Adapter layer-4: Domain norm
        mean = output.mean(dim=0)
        var = output.var(dim=0)
        output_norm = (output - mean) / torch.sqrt(var + self.eps)
        output = self.gamma * output_norm

        self.omics_out[omics_id] = output

        # Residual connection
        return self.residual + output