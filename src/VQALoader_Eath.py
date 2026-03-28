import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image


QUESTIONS = ['Are there any villages in this scene?', 'Is there a commercial area near the residential area?',
             'Are there any playgrounds in this scene?', 'Is there any commercial land in this scene?',
             'Is there any forest in this scene?', 'Is there any agriculture in this scene?',
             'What are the types of residential buildings?', 'Are there any urban villages in this scene?',
             'What are the needs for the renovation of villages?', 'Is there any barren in this scene?',
             'Whether greening need to be supplemented in residential areas?', 'Is there any woodland in this scene?',
             'What are the land use types in this scene?', 'What are the needs for the renovation of residents?',
             'Are there any buildings in this scene?', 'Is there any agricultural land in this scene?',
             'What is the area of roads?', 'What is the area of playgrounds?', 'Is it a rural or urban scene?',
             'What is the area of barren?', 'Are there any bridges in this scene?',
             'Are there any eutrophic waters in this scene?', 'Are there any viaducts in this scene?',
             'What is the area of water?', 'Are there any roads in this scene?',
             'Is there any residential land in this scene?', 'How many eutrophic waters are in this scene?',
             'Is there any industrial land in this scene?', 'Is there any park land in this scene?',
             'Is there any uncultivated agricultural land in this scene?',
             'Is there a school near the residential area?', 'Are there any large driveways (more than four lanes)?',
             'What are the comprehensive traffic situations in this scene?', 'What is the area of buildings?',
             'Is there any construction land in this scene?', 'What are the water types in this scene?',
             'Are there any viaducts near the residential area?',
             'Is there a construction area near the residential area?', 'Is there a park near the residential area?',
             'What are the road materials around the village?', 'Are there any intersections in this scene?',
             'What are the road types around the residential area?',
             'What are the water situations around the agricultural land?', 'What is the situation of barren land?',
             'What is the area of the forest?', 'Are there any intersections near the school?',
             'Is there any water in this scene?', 'Is there any educational land in this scene?',
             'How many intersections are in this scene?', 'Are there any greenhouses in this scene?',
             'What is the area of agriculture?']
QUESTION_TYPES = ['Basic Judging', 'Reasoning-based Judging', 'Basic Counting', 'Reasoning-based Counting',
                  'Object Situation Analysis', 'Comprehensive Analysis']


class EarthVQADataset(Dataset):
    """
    EarthVQA 数据集加载器
    Args:
        json_path: JSON 文件路径
        img_folder: 图像文件夹路径
        answer_voc: 候选答案列表 (ANSWER_VOC)
        tokenizer: 文本 tokenizer
        image_processor: 图像处理器
        seq_len: 语言序列长度
        img_transform: 图像预处理，可选
    """

    def __init__(self, json_path, img_folder, answer_voc, tokenizer, image_processor, seq_len=40, img_transform=None):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        self.img_folder = img_folder
        self.answer_voc = answer_voc
        self.ans2idx = {ans: idx for idx, ans in enumerate(self.answer_voc)}
        self.idx2ans = {idx: ans for ans, idx in self.ans2idx.items()}
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.seq_len = seq_len
        self.img_transform = img_transform

        # 把 JSON 转成列表，每个元素是一个 QA 对
        self.items = []
        for img_name, qa_list in self.annotations.items():
            for qa in qa_list:
                self.items.append({
                    "image_name": img_name,
                    "question": qa["Question"],
                    "question_type": qa["Type"],
                    "answer": qa["Answer"]
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # 图像路径
        img_path = os.path.join(self.img_folder, item["image_name"])
        image = Image.open(img_path).convert('RGB')
        if self.img_transform:
            image = self.img_transform(image)
        imgT = self.image_processor(image, return_tensors="pt")
        pixel_values = imgT['pixel_values'][0]

        # 问题编码
        question = item["question"]
        question_feats = self.tokenizer(
            question,
            return_tensors='pt',
            padding='max_length',
            max_length=self.seq_len
        )
        input_ids = question_feats['input_ids'][0]
        attention_mask = question_feats['attention_mask'][0]

        # 问题类型编码
        qtype = item["question_type"]
        qtype_feats = self.tokenizer(
            qtype,
            return_tensors='pt',
            padding='max_length',
            max_length=self.seq_len
        )
        label = qtype_feats['input_ids'][0]
        label_attention_mask = qtype_feats['attention_mask'][0]

        # 答案编码
        ans_str = item["answer"]
        ans_label = self.ans2idx.get(ans_str, 0)
        q_idx = QUESTIONS.index(question)
        image_n = item["image_name"]
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            # "question": question,
            "q_idx": q_idx,
            "image_n": image_n,
            "question_type": qtype,
            "labels": label,
            "label_attention_mask": label_attention_mask,
            "answer": torch.tensor(ans_label, dtype=torch.long),
            # "answer_str": ans_str
        }
