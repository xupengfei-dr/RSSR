import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import BlipImageProcessor, BlipProcessor

import torchvision.transforms as transforms

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
        ans_label = self.ans2idx.get(ans_str, 0)  # 如果答案不在 VOC 中，映射为 0

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "question": question,
            "question_type": qtype,
            "labels": label,
            "label_attention_mask": label_attention_mask,
            "answer": torch.tensor(ans_label, dtype=torch.long),
            "answer_str": ans_str
        }

ANSWER_VOC = [0, 1, 2, 3, 4, 5, 6, '0%-10%', '10%-20%', '20%-30%', '30%-40%', '40%-50%', '50%-60%', '60%-70%',
              '70%-80%', '80%-90%', '90%-100%', 'The roads need to be improved, and waters need to be cleaned up',
              'This is an important traffic area with 3 intersections',
              'There are residential, educational, park, and agricultural areas', 'Developing', 'There are railways',
              'This is a very important traffic area with 1 intersection, several viaducts, and several bridges',
              'There are cement roads', 'There are educational, construction, and agricultural areas', 'Underdeveloped',
              'There are unsurfaced roads, and cement roads',
              'There are residential, commercial, park, and agricultural areas', 'There are commercial areas',
              'This is a very important traffic area with 2 intersections, and several viaducts',
              'There are commercial, construction, and park areas',
              'There are residential, commercial, park, industrial, and agricultural areas',
              'There are commercial, and construction areas', 'This is not an important traffic area',
              'This is a very important traffic area with 2 intersections, and several bridges',
              'There are unsurfaced roads, and railways', 'There are woodland, industrial, and agricultural areas',
              'There are park areas', 'There are construction, park, and agricultural areas',
              'There are residential, and industrial areas', 'There are residential, and construction areas',
              'There is no water', 'There are residential, construction, and park areas',
              'There are commercial buildings', 'There are agricultural areas', 'There are educational areas',
              'There are residential, and commercial areas',
              'There are commercial, educational, park, and industrial areas',
              'There are clean waters near the agriculture land', 'There are ponds',
              'There are residential, commercial, park, and industrial areas',
              'There are educational, park, industrial, and agricultural areas',
              'There are unsurfaced roads, cement roads, railways, and asphalt roads',
              'There are one-way lanes, and railways',
              'There are residential, commercial, educational, park, and industrial areas', 'There are no water area',
              'There are railways, and asphalt roads', 'There are construction areas',
              'The urban villages need attention', 'There are unsurfaced roads, railways, and asphalt roads',
              'There are residential, and agricultural areas',
              'There are residential, commercial, and agricultural areas', 'No',
              'This is a very important traffic area with 1 intersection, and several viaducts',
              'The greening needs to be supplemented',
              'There are residential, commercial, educational, and construction areas',
              'This is an important traffic area with several bridges',
              'There are residential, commercial, educational, and industrial areas', 'There are woodland areas',
              'There are residential, commercial, and construction areas', 'Rural',
              'There are residential, construction, park, industrial, and agricultural areas',
              'There are residential, woodland, industrial, and agricultural areas',
              'This is an important traffic area with 4 intersections', 'There are private buildings',
              'There are woodland, and agricultural areas',
              'There are residential, commercial, construction, and park areas', 'There are rivers and ponds',
              'There are residential, construction, and agricultural areas',
              'There are residential, and educational areas', 'There are commercial, and educational areas',
              'There are polluted waters near the agriculture land',
              'There are one-way lanes, wide lanes, and railways', 'There are one-way lanes, and wide lanes', 'Urban',
              'There are residential, commercial, and educational areas', 'There are commercial, and park areas',
              'There are unsurfaced roads, cement roads, and asphalt roads',
              'There are commercial buildings, and private buildings',
              'This is an important traffic area with 1 intersection',
              'There are commercial, industrial, and agricultural areas',
              'There are residential, commercial, construction, park, and industrial areas', 'There are asphalt roads',
              'There are residential, commercial, and park areas', 'There are no agricultural land',
              'There are commercial, construction, park, and agricultural areas',
              'There are residential, educational, and construction areas',
              'There are commercial, construction, and industrial areas',
              'There are residential, commercial, construction, and industrial areas',
              'There are park, and industrial areas', 'There are commercial, and agricultural areas',
              'There are residential, educational, construction, and park areas', 'No obvious land use types',
              'There are construction, park, and industrial areas',
              'There are residential, educational, park, and industrial areas',
              'There are commercial, park, and industrial areas',
              'This is an important traffic area with several viaducts',
              'This is a very important traffic area with 1 intersection, and several bridges',
              'There are residential, park, and agricultural areas',
              'There are residential, commercial, construction, and agricultural areas',
              'There are residential, commercial, educational, construction, park, and agricultural areas',
              'There are wide lanes, and railways', 'There are residential, park, and industrial areas',
              'There are residential, industrial, and agricultural areas', 'There are construction, and park areas',
              'There are residential, commercial, construction, park, industrial, and agricultural areas',
              'There are residential, park, industrial, and agricultural areas', 'There are residential areas',
              'There are residential, commercial, educational, park, and agricultural areas',
              'There are residential, commercial, industrial, and agricultural areas',
              'There are residential, commercial, educational, and park areas',
              'There are construction, and agricultural areas', 'There are no water nor agricultural land',
              'The waters need to be cleaned up', 'There are park, and agricultural areas', 'There are rivers',
              'This is a very important traffic area with 3 intersections, and several viaducts',
              'This is an important traffic area with 2 intersections', 'There are industrial areas',
              'There are unsurfaced roads, and asphalt roads',
              'This is a very important traffic area with 2 intersections, several viaducts, and several bridges',
              'There are commercial, park, and agricultural areas', 'There are one-way lanes',
              'There are residential, educational, construction, and agricultural areas', 'There are no roads',
              'There are residential, construction, park, and agricultural areas',
              'There are residential, and park areas', 'There are commercial, construction, and agricultural areas',
              'There are cement roads, and asphalt roads', 'There are residential, educational, and agricultural areas',
              'There are commercial, and industrial areas', 'There are park, industrial, and agricultural areas',
              'This is a very important traffic area with several viaducts, and several bridges',
              'There are educational, construction, and park areas',
              'There are residential, woodland, and agricultural areas', 'There are residential, and woodland areas',
              'There are unsurfaced roads, cement roads, and railways',
              'There are educational, park, and agricultural areas',
              'There are residential, educational, and park areas', 'There are commercial, educational, and park areas',
              'There are wide lanes', 'There are cement roads, and railways', 'There are no residential buildings',
              'There are commercial, park, industrial, and agricultural areas',
              'There are residential, commercial, and industrial areas',
              'The greening needs to be supplemented and urban villages need attention', 'There is no barren land',
              'There are educational, and agricultural areas', 'The roads need to be improved', 'Yes',
              'There are unsurfaced roads',
              'There are residential, commercial, construction, park, and agricultural areas',
              'There are residential, construction, and industrial areas',
              'There are cement roads, railways, and asphalt roads', 'There are educational, and park areas',
              'There are no needs']
image_processor = BlipImageProcessor(do_resize=True, image_std=[0.229, 0.224, 0.225], image_mean=[0.485, 0.456, 0.406],
                                     do_rescale=True, do_normalize=True, size=256, size_divisor=32)
processor = BlipProcessor.from_pretrained("/home/gpuadmin/blip-vqa-base")
tokenizer = processor.tokenizer

transform_train = [
    transforms.RandomHorizontalFlip(),
]
# 示例：训练集和验证集
train_dataset = EarthVQADataset(
    json_path='/home/gpuadmin/DataSet/EathVQA/Train_QA.json',
    img_folder='/home/gpuadmin/DataSet/EathVQA/Train/images_png',
    answer_voc=ANSWER_VOC,
    tokenizer=tokenizer,
    image_processor=image_processor
)

val_dataset = EarthVQADataset(
    json_path='/home/gpuadmin/DataSet/EathVQA/Val_QA.json',
    img_folder='/home/gpuadmin/DataSet/EathVQA/Val/images_png',
    answer_voc=ANSWER_VOC,
    tokenizer=tokenizer,
    image_processor=image_processor
)
# 输出训练集前 3 条样本
for i in range(3):
    sample = train_dataset[i]
    print(f"Sample {i}:")
    print("Question:", sample['question'])
    print("Question Type:", sample['question_type'])
    print("Answer:", sample['answer_str'], "-> Label:", sample['answer'])
    print("Pixel Values Shape:", sample['pixel_values'].shape)
    print("Input IDs Shape:", sample['input_ids'].shape)
    print("Attention Mask Shape:", sample['attention_mask'].shape)
    print("Labels Shape:", sample['labels'].shape)
    print("Label Attention Mask Shape:", sample['label_attention_mask'].shape)
    print("="*50)

# 同理可以输出验证集前 3 条
for i in range(3):
    sample = val_dataset[i]
    print(f"Val Sample {i}:")
    print("Question:", sample['question'])
    print("Question Type:", sample['question_type'])
    print("Answer:", sample['answer_str'], "-> Label:", sample['answer'])
    print("Pixel Values Shape:", sample['pixel_values'].shape)
    print("Input IDs Shape:", sample['input_ids'].shape)
    print("Attention Mask Shape:", sample['attention_mask'].shape)
    print("Labels Shape:", sample['labels'].shape)
    print("Label Attention Mask Shape:", sample['label_attention_mask'].shape)
    print("="*50)