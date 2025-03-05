import torch
import numpy as np


def element_wise_subtraction(a, b):
    return a - b


def element_wise_multiplication(a, b):
    return a * b


def element_wise_division(a, b):
    return a / b


def concatenate(a, b):
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    return torch.cat([a, b], dim=1)


def average(a, b):
    return (a + b) / 2


def attention(a, b):
    attention_weights = torch.nn.functional.softmax(torch.tensor([0.5, 0.5]), dim=0)
    return a * attention_weights[0] + b * attention_weights[1]


def linear_transformation(a, b):
    linear = torch.nn.Linear(512, 512)
    fused = torch.cat((linear(a), linear(b)), dim=0)
    return linear(fused)


def cbp(a, b, output_dim=8000):
    from CompactBilinearPooling import CompactBilinearPooling
    layer = CompactBilinearPooling(512, 512, 8000)
    layer.train()
    out = layer(a, b)
    return out
