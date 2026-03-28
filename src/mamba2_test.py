# text_classification.py
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from datasets import load_dataset

from mamba_ssm.modules.mamba2_simple import Mamba2Simple  # 引用完整的模型


class TextClassifier(nn.Module):
    def __init__(self, d_model, num_classes):
        super(TextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.mamba_model = Mamba2Simple(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, d_model)
        out = self.mamba_model(hidden_states)  # 使用 Mamba2Simple 模型
        out = self.fc(out[:, 0, :])  # 取 [CLS] 标记的输出
        return out


def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def main():
    # 加载数据集
    dataset = load_dataset('imdb')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenization
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=8)

    # 模型参数
    d_model = 768  # BERT的隐藏维度
    num_classes = 2  # IMDB是二分类
    model = TextClassifier(d_model, num_classes)

    # 训练设置
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # 训练模型
    train_model(model, train_dataloader, criterion, optimizer, device)


if __name__ == "__main__":
    main()
