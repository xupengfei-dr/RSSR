# RSAdapter
import torch
from torch import nn



class MyAdapter(nn.Module):
    def __init__(self, D_features, D_dim=192, act_layer=nn.GELU, red_connect=False):
        super().__init__()
        self.red_connect = red_connect
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_dim)
        self.D_fc2 = nn.Linear(D_dim, D_features)
        self.weight_1, self.bias_1 = self.init_AT(D_dim)
        self.weight_2, self.bias_2 = self.init_AT(D_features)

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.AT(xs, self.weight_1, self.bias_1)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        xs = self.AT(xs, self.weight_2, self.bias_2)

        if self.red_connect:
            x = x + xs
        else:
            x = xs

        return x

    def AT(self, x, weight, bias):
        return x * weight + bias

    def init_AT(self, dim):
        weight = nn.Parameter(torch.ones(dim))
        bias = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(weight, mean=1, std=.02)
        nn.init.normal_(bias, std=.02)
        return weight, bias
