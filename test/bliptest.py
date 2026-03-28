import numpy as np
import requests
import torch
from PIL import Image
from transformers import SiglipImageProcessor, AutoTokenizer, AutoModel

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

from src.trans.models.blip.modeling_blip_test2 import BlipForQuestionAnswering

if __name__ == '__main__':
    path = "/home/pengfei/blip-vqa-capfilt-large"
    siglip_path = "/home/pengfei/siglip-base-patch16-224"

    model = BlipForQuestionAnswering.from_pretrained(path).to(device)


    # vm = AutoModel.from_pretrained(siglip_path).to(device)

    image_processor = SiglipImageProcessor.from_pretrained(siglip_path)

    text = "A cat is sitting on a sofa."
    tokenizer = AutoTokenizer.from_pretrained(siglip_path)

    inputs = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True,max_length=40,return_attention_mask=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    image = np.random.rand(1, 3, 224, 224)  # 生成随机图像
    pixel_values = torch.tensor(image, dtype=torch.float).to(device)


    outputs = model(
        input_ids=input_ids,  # 文本输入
        pixel_values=pixel_values,  # 图像输入
        output_attentions=True,  # 输出 attention 权重 (可选)
        output_hidden_states=True,  # 输出隐藏状态 (可选)
        return_dict=True,
        attention_mask=attention_mask,
        labels= ""
    )
    vision_model_output = outputs.vision_model_output
    print(vision_model_output.last_hidden_state.shape)
    exit()

    # img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    # raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    #
    # question = "how many dogs are in the picture?"
    # inputs = processor(raw_image, question, return_tensors="pt")
    #         fused_features = self.multi_modal_projector(image_features)

    #  self.multi_modal_projector = MultiModalProjector(
    #             vision_hidden_dim=config.vision_config.hidden_size,
    #             text_hidden_dim=config.text_config.hidden_size
    #             # config.text_config.hidden_size,  #  text_config 如果为None,就随意传入一个int
    #         )
    # class MultiModalProjector(nn.Module):
    #     def __init__(self, vision_hidden_dim, text_hidden_dim):
    #         super().__init__()
    #         self.linear_1 = nn.Linear(vision_hidden_dim, text_hidden_dim)
    #         self.act = nn.GELU()
    #         self.linear_2 = nn.Linear(text_hidden_dim, text_hidden_dim)
    #
    #     def forward(self, image_features):
    #         hidden_states = self.linear_1(image_features)
    #         hidden_states = self.act(hidden_states)
    #         hidden_states = self.linear_2(hidden_states)
    #         return hidden_states

    # text = "A cat is sitting on a sofa."
    #
    # # 编码文本为 input_ids
    # inputs = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True,max_length=40,return_attention_mask=True)
