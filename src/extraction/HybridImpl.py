import torch
import numpy as np
from HybridBase import Hybrid


class Minus(Hybrid):
    @staticmethod
    def hybrid_features(a, b, **kwargs):
        c = a - b
        return c


class Multiplication(Hybrid):
    @staticmethod
    def hybrid_features(a, b, **kwargs):
        return a * b


class Division(Hybrid):
    @staticmethod
    def hybrid_features(a, b, **kwargs):
        return a / b


class Concatenate(Hybrid):
    @staticmethod
    def hybrid_features(a, b, **kwargs):
        return np.concatenate((a, b), axis=1)


class Average(Hybrid):
    @staticmethod
    def hybrid_features(a, b, **kwargs):
        return (a + b) / 2


class Attention(Hybrid):
    @staticmethod
    def hybrid_features(a, b, **kwargs):
        attention_img = kwargs.get('img', 0.7)
        attention_text = kwargs.get('text', 0.3)
        attention_weights = torch.nn.functional.softmax(torch.tensor([attention_img, attention_text]), dim=0)
        attention_weights = attention_weights.cpu().detach().numpy()
        return a * attention_weights[0] + b * attention_weights[1]


class LinearTransformation(Hybrid):
    @staticmethod
    def hybrid_features(a, b, **kwargs):
        linear = torch.nn.Linear(512, 512)
        fused = torch.cat((linear(a), linear(b)), dim=0)
        return linear(fused)


class CBP(Hybrid):
    @staticmethod
    def hybrid_features(a, b, **kwargs):
        from HybridUtils import CompactBilinearPooling
        output_dim = 8000
        layer = CompactBilinearPooling(a.size, b.size, output_dim)
        layer.train()
        a = torch.tensor(a, dtype=torch.float32)
        b = torch.tensor(b, dtype=torch.float32)
        out = layer(a, b)
        out = torch.nn.functional.normalize(out, p=2, dim=1)
        return out.numpy()
