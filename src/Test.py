from transformers import BertModel, BertTokenizer

# model2 = BertModel.from_pretrained('./bert-base-chinese')
# # model=BertTokenizer.from_pretrained('./bert-base-chinese')
# print(model2)

if __name__ == '__main__':
    import torch

    # 加载 .ckpt 文件
    checkpoint = torch.load('/home/pengfei/rsvqa/src/train_models_inner_4e5_dim192_mamba/epoch=0_valid_acc=0.85527_base_myencoder.ckpt')
    # 提取 state_dict
    state_dict = checkpoint['state_dict']

    # 打印 state_dict 中所有参数的名字和形状
    for name, param in state_dict.items():
        print(f"Parameter name: {name}, Shape: {param.shape}")
