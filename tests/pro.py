import torch
import torch.nn as nn

# 原始数据
answers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
prd = [1, 2, 2, 4, 5, 7, 7, 8, 9]

# 嵌入维度
embedding_dim = 768

# 创建一个嵌入层
# 假设类别标签从 0 开始，因此需要将数组中的值减 1
embedding_layer = nn.Embedding(num_embeddings=10, embedding_dim=embedding_dim)  # num_embeddings = 最大类别数 + 1

# 将数组转换为张量，并减去 1（因为嵌入层的索引从 0 开始）
answers_tensor = torch.tensor(answers, dtype=torch.long) - 1
prd_tensor = torch.tensor(prd, dtype=torch.long) - 1

# 使用嵌入层将每个类别标签映射为 768 维的向量
answers_embedded = embedding_layer(answers_tensor)
prd_embedded = embedding_layer(prd_tensor)

# 检查形状
print("Shape of answers_embedded:", answers_embedded.shape)  # 应该是 [9, 768]
print("Shape of prd_embedded:", prd_embedded.shape)  # 应该是 [9, 768]

# 打印前几行以确认内容
print("First few rows of answers_embedded:\n", answers_embedded[:3])
print("First few rows of prd_embedded:\n", prd_embedded[:3])