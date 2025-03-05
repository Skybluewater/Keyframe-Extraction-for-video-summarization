import torch
import torch.nn as nn

# 创建两个嵌入向量
embedding1 = torch.tensor([[1.0, 2.0, 3.0]])
embedding2 = torch.tensor([[4.0, 5.0, 6.0]])

# 添加 Multimodal Compact Bilinear Pooling (MCB) 融合函数
def mcb_pooling(x, y, output_dim):
    # 输入: x: [batch, dim1], y: [batch, dim2]
    batch_size, d1 = x.size()
    d2 = y.size(1)
    h1 = torch.randint(output_dim, (d1,), dtype=torch.long)
    s1 = torch.randint(2, (d1,), dtype=torch.float32) * 2 - 1
    h2 = torch.randint(output_dim, (d2,), dtype=torch.long)
    s2 = torch.randint(2, (d2,), dtype=torch.float32) * 2 - 1

    sketch1 = torch.zeros(batch_size, output_dim)
    sketch2 = torch.zeros(batch_size, output_dim)

    for i in range(d1):
        sketch1[:, h1[i]] += s1[i] * x[:, i]
    for i in range(d2):
        sketch2[:, h2[i]] += s2[i] * y[:, i]

    fft1 = torch.fft.rfft(sketch1)
    fft2 = torch.fft.rfft(sketch2)
    fft_product = fft1 * fft2
    cbp = torch.fft.irfft(fft_product, n=output_dim)
    return cbp

# 使用 MCB 融合两个嵌入向量
fused_embedding = mcb_pooling(embedding1, embedding2, output_dim=8)
print(fused_embedding)


from compact_bilinear_pooling import CountSketch, CompactBilinearPooling

input_size = 2048
output_size = 16000
mcb = CompactBilinearPooling(input_size, input_size, output_size)
x = torch.rand(4,input_size)
y = torch.rand(4,input_size)

z = mcb(x,y)