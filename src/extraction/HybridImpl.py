import torch
import numpy as np
from HybridBase import Hybrid


class Subtraction(Hybrid):
    @staticmethod
    def hybrid_featurs(a, b):
        return a - b


class Multiplication(Hybrid):
    @staticmethod
    def hybrid_features(a, b):
        return a * b


class Division(Hybrid):
    @staticmethod
    def hybrid_features(a, b):
        return a / b


class Concatenate(Hybrid):
    @staticmethod
    def hybrid_features(a, b):
        return np.concatenate((a, b), axis=1)


class Average(Hybrid):
    @staticmethod
    def hybrid_features(a, b):
        return (a + b) / 2


class Attention(Hybrid):
    @staticmethod
    def hybrid_features(a, b):
        attention_weights = torch.nn.functional.softmax(torch.tensor([0.5, 0.5]), dim=0)
        return a * attention_weights[0] + b * attention_weights[1]


class LinearTransformation(Hybrid):
    @staticmethod
    def hybrid_features(a, b):
        linear = torch.nn.Linear(512, 512)
        fused = torch.cat((linear(a), linear(b)), dim=0)
        return linear(fused)


class CBP(Hybrid):
    @staticmethod
    def hybrid_features(a, b):
        from extraction.HybridUtils import CompactBilinearPooling
        output_dim = 8000
        layer = CompactBilinearPooling(512, 512, output_dim)
        layer.train()
        out = layer(a, b)
        return out
