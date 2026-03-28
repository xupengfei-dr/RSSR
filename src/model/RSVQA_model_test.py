import json
import os
import sys
import torch
import numpy as np
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import sys

# from MyClassify import EnhancedClassifier
from src.model.CLIP_QWEN_CO import VLQ,MyVLQConfig
from src.trans import SiglipModel

sys.path.append("../../")
sys.path.append("../")


# from src.t.src.transformers.models.blip.modeling_blip_test import BlipForQuestionAnswering, BlipModel
# from src.t.src.transformers.models.blip.modeling_blip_test import BlipForQuestionAnswering, BlipModel
# from src.t.src.transformers.models.blip.modeling_blip_test import BlipForQuestionAnswering, BlipModel
from src.trans.models.blip.modeling_blip_test  import BlipForQuestionAnswering,BlipModel
from transformers import BlipConfig
from src.model.MyEncoder import MyEncoder
from src.model.MyUtils import In2,In2Conv1d,GlobalQueryAttentionPool
from src.model.PromptUtils import AnswerConditionalClassifier,AnswerConditionalClassifierWithFiLM
# from src.model.MyMamba import Mamba2Model,Mamba2Config
from src.myMambaTest import Mamba2Config,Mamba2Model
path = ''
# qwen_model_path = "/home/pengfei/Qwen2.5-1.5B-Instruct"
# clip_vit_model_Path = "/home/pengfei/clip-vit-large-patch14-336"

# config = MyVLQConfig(vision_path=clip_vit_model_Path, text_path=qwen_model_path)

class VQAModel(pl.LightningModule):
    def __init__(self, batch_size=None, lr=None, number_outputs=None):
        super(VQAModel, self).__init__()

        self.save_hyperparameters()
        self.number_outputs = number_outputs
        self.loss = F.cross_entropy
        self.lr = lr
        self.batch_size = batch_size
        self.validation_step_outputs = []
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=number_outputs)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=number_outputs)
        self.results = {}
        self.res = []
        # self.clipqwen = VLQ(config)
        # self.mambaconfig =Mamba2Config()
        # self.mamba = Mamba2Model(self.mambaconfig)

        # self.blip = BlipModel.from_pretrained("Salesforce/blip-vqa-base")
        self.blip = BlipForQuestionAnswering.from_pretrained("/home/pengfei/blip-vqa-base")
        # self.siglip_model = SiglipModel.from_pretrained("/home/pengfei/siglip2-large-patch16-256")
        # self.blip = BlipForQuestionAnswering.from_pretrained("/home/pengfei/rsvqa/src/Salesforce/blip-vqa-base")

        # self.myencoder = MyEncoder(BlipConfig.from_dict(json.load(open('model/layer_config.json'))))

        self.in2 = In2()
        # self.yindaoacc = AnswerConditionalClassifier(768,self.number_outputs,8)
        # self.menkong = WeakAnswerConditionalClassifier(768,self.number_outputs,8)
        # self.in2conv1d = In2Conv1d()
        # self.in2ga = GlobalQueryAttentionPool()
        # self.classify = EnhancedClassifier(768,9)
        # self.filmacc = AnswerConditionalClassifierWithFiLM(768,self.number_outputs,1,0.98)
        #conattention   multi_modal_projector
        for name, param in self.blip.named_parameters():
            # if 'adapter' not in name and 'text_encoder.' not in name:
            if 'adapter' not in name and 'text_encoder.' not in name:
                param.requires_grad = False
                print(name)
            # param.requires_grad = False
            # if 'adapter' not in name and 'text_encoder' not in name and 'vision_model.embeddings' not in name and 'vision_model.post_layernorm' not in name:
            #     # if 'vision_model.encoder.layers.0' not in name and 'vision_model.encoder.layers.1' not in name and'vision_model.encoder.layers.10' not in name and'vision_model.encoder.layers.11' not in name \
            #     #         and'vision_model.encoder.layers.5' not in name and'vision_model.encoder.layers.6' not in name and'vision_model.encoder.layers.7' not in name:


        self.classify_layer = nn.Sequential(
            nn.Linear(768, 1024),
            nn.BatchNorm1d(1024),
            torch.nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            torch.nn.GELU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            torch.nn.GELU(),
            nn.Linear(256, self.number_outputs)
        )


    def forward(self, pixel_values, input_ids, attention_mask, labels, labels_attention_mask):

        # print(self.siglip_model.vision_model)

        # self.blip.vision_model=self.siglip_model.vision_model
        # print(self.blip.vision_model)
        # exit()
        out = self.blip(pixel_values=pixel_values, input_ids=input_ids,
                        attention_mask=attention_mask, labels=labels
                        )

        # loss =out['loss']
        # 从 BLIP 输出中获取文本嵌入
        # 格式为 64，257，768
        result = torch.squeeze(out['last_hidden_state_vero'])
        # img_embedding = out['image_embeds']
        # input_embedding = torch.concat([question_embedding,img_embedding],dim=1)
        # input_embedding_attention_mask= out['input_embedding_attention_mask']
        # out2 = torch.squeeze(out['last_hidden_state_encoder'][:, 0, :])
        # todo:对blip输出的东西进行编码层处理

        # todo:---------------------压缩特征-----------------64,40,768->64,768->64,10
        # result = (self.encoder(inputs_embeds=question_embedding))['last_hidden_state'][:, 0, :]
        # todo:xxxxxxxxxxx消融
        # result = (self.myencoder(inputs_embeds=result))['last_hidden_state']

        res = self.in2(result)
        # res = self.in2(question_embedding)
        # res = result[:,0,:]


        # todo:--------------------------------------
        # todo:--------------------引导答案-------------------
        # logist = self.filmacc(res,indexacc)
        logist = self.classify_layer(res)
        # logist = self.classify_layer(result)

        # todo：------------------------------------------------

        # out2 = self.classify_layer(res)



        # todo:内嵌了，直接使用
        # last_state = out['vision_model_output']['last_hidden_state']
        # print(last_state.shape)
        # --------------------------
        # out = torch.squeeze(question_embedding[:, 0, :])
        # 获得预测结果

        # ra = question_embedding
        # todo:内嵌后再加myencoder
        # out = (self.encoder(inputs_embeds=question_embedding))['last_hidden_state']
        # out = (ra + out)[:, 0, :]
        # todo:mamba
        # result = (self.mamba(inputs_embeds=question_embedding))[0]
        # result=result[:, 0, :]
        # result = (self.mamba(inputs_embeds=question_embedding))[0][:, 0, :]


        #todo: ----------------
        # result = self.clipqwen(pixel_values, input_ids, attention_mask, labels=labels)
        # res = self.in2(result)
        # logist = self.classify_layer(res)



        return logist

    def configure_optimizers(self):
        # configuration of the optimizer
        def rule(epoch):
            if self.number_outputs == 9:
                if epoch <= 3:
                    lamda = 1
                else:
                    lamda = 0.01
                return lamda
            else:

                # 3 1  ,10 0.1 0.06
                if epoch <= 3:
                    lamda = 1 + epoch
                elif epoch <= 6:
                    lamda = 0.08
                else:
                    lamda = 0.01
                return lamda
            # else:
            #     if epoch <= 2:
            #         lamda = 1
            #     elif epoch <= 6:  #10
            #         lamda = 0.08
            #     else:
            #         lamda = 0.01
            #     return lamda

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        # optimizer = torch.optim.AdamW()
        scheduler = LambdaLR(optimizer, lr_lambda=rule)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def training_step(self, batch, batch_idx):
        # performs the training steps

        # pixel_values, input_ids,  attention_mask, labels,answer = batch
        # 推理时解包 batch
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        # tishi_language_feats = batch["tishi_language_feats"]
        # tishi_attention_mask = batch["tishi_attention_mask"]

        # print(tishi_language_feats)
        # exit()
        # print("label",labels)
        # exit()
        answer = batch["answer"]

        # print(answer,'---------------------------------------------answer')
        # print(answer.shape,'---------------------------------------------answer')
        # exit()
        # print(answer,"answer----------------------")
        labels_attention_mask = batch["label_attention_mask"]
        pred = self(pixel_values, input_ids, attention_mask, labels, labels_attention_mask)
        # pred = self(pixel_values, tishi_language_feats, tishi_attention_mask, labels, labels_attention_mask)
        # print(answer,"pred----------------------")
        # exit()
        self.train_acc(pred, answer)
        train_loss = self.loss(pred, answer)
        self.log("train_loss", train_loss, on_epoch=True, on_step=True, sync_dist=True, batch_size=self.batch_size)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=True, sync_dist=True, batch_size=self.batch_size)

        return train_loss

    def validation_step(self, batch, batch_idx):
        # pixel_values, input_ids, token_type_ids, attention_mask, answer, question_type, img_id, question, answer_str = batch
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        answer = batch["answer"]
        question_type = batch["question_type"]
        labels = batch["labels"]
        labels_attention_mask = batch["label_attention_mask"]
        # tishi_language_feats = batch["tishi_language_feats"]
        # tishi_attention_mask = batch["tishi_attention_mask"]

        # inputs = {
        #     "pixel_values": pixel_values,
        #     "input_ids": input_ids,
        #     "attention_mask": attention_mask,
        #     "labels": labels,
        #     "labels_attention_mask": labels_attention_mask
        # }


        pred = self(pixel_values, input_ids, attention_mask, labels, labels_attention_mask)
        # pred = self(pixel_values, tishi_language_feats, tishi_attention_mask, labels, labels_attention_mask)

        self.valid_acc(pred, answer)
        valid_loss = self.loss(pred, answer)

        self.log("valid_loss", valid_loss, on_epoch=True, on_step=False, sync_dist=True, batch_size=self.batch_size)
        self.log("valid_acc", self.valid_acc, on_epoch=True, on_step=False, sync_dist=True, batch_size=self.batch_size)

        pred_arg = torch.argmax(pred, axis=1)
        for i in range(pred.shape[0]):
            if pred_arg[i] == answer[i]:
                self.validation_step_outputs.append([1, question_type[i]])
            else:
                self.validation_step_outputs.append([0, question_type[i]])

    def on_validation_epoch_end(self):
        outputs = np.stack(self.validation_step_outputs)

        total_rural_urban, total_presence, total_count, total_comp = 0, 0, 0, 0
        right_rural_urban, right_presence, right_count, right_comp = 0, 0, 0, 0
        acc_rural_urban, acc_presence, acc_count, acc_comp = 0, 0, 0, 0
        AA, OA, right, total = 0, 0, 0, 0

        for i in range(outputs.shape[0]):
            if outputs[i][1] == 'comp':
                total_comp += 1
                if outputs[i][0] == '1':
                    right_comp += 1
            elif outputs[i][1] == 'presence':
                total_presence += 1
                if outputs[i][0] == '1':
                    right_presence += 1
            elif outputs[i][1] == 'count':
                total_count += 1
                if outputs[i][0] == '1':
                    right_count += 1
            else:
                total_rural_urban += 1
                if outputs[i][0] == '1':
                    right_rural_urban += 1

        # Note that for RSVQA_HR, there's no 'rural_urban' question type 
        # so 'rural_urban' in RSVQA_HR represent for 'area' question type
        acc_rural_urban = right_rural_urban / total_rural_urban
        acc_presence = right_presence / total_presence
        acc_count = right_count / total_count
        acc_comp = right_comp / total_comp

        right = right_rural_urban + right_presence + right_count + right_comp
        total = total_rural_urban + total_presence + total_count + total_comp

        AA = (acc_rural_urban + acc_presence + acc_count + acc_comp) / 4
        OA = right / total

        self.log("acc_rural_urban", acc_rural_urban, sync_dist=True)
        self.log("acc_presence", acc_presence, sync_dist=True)
        self.log("acc_count", acc_count, sync_dist=True)
        self.log("acc_comp", acc_comp, sync_dist=True)
        # self.log("total", total, sync_dist=True)
        self.log('valid_AA', AA, sync_dist=True)
        self.log('valid_OA', OA, sync_dist=True)

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        # Unpack the batch data
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        # tishi_language_feats = batch["tishi_language_feats"]
        answer = batch["answer"]
        questions = batch["question"]

        # print(questions[359])
        # exit()
        labels_attention_mask = batch["label_attention_mask"]
        question_type = batch['question_type']

        pred = self(pixel_values, input_ids, attention_mask, labels, labels_attention_mask)
        # pred = self(pixel_values, tishi_language_feats, attention_mask, labels, labels_attention_mask)

        # Compute the test loss
        test_loss = self.loss(pred, answer)

        # Compute the test accuracy
        test_acc = (pred.argmax(dim=1) == answer).float().mean()

        # Compute accuracy for each question type (e.g., yes/no)
        question_type_acc = {}
        for q_type in set(question_type):  # Iterate over unique question types
            indices = [i for i, q in enumerate(question_type) if q == q_type]
            if indices:  # If there are any examples for this question type
                q_type_preds = pred[indices].argmax(dim=1)
                q_type_answers = answer[indices]
                q_type_acc = (q_type_preds == q_type_answers).float().mean().item()
                question_type_acc[q_type] = round(q_type_acc * 100, 2)

        # Log the overall metrics
        self.log("test_loss", test_loss, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        self.log("test_acc", test_acc, on_epoch=True, sync_dist=True, batch_size=self.batch_size)

        # Log per-question-type accuracy (as numeric values, not formatted strings)
        for q_type, acc in question_type_acc.items():
            self.log(f"test_acc_{q_type}", acc, on_epoch=True, sync_dist=True, batch_size=self.batch_size)

        # Log the formatted accuracy (for display purposes)
        for q_type, acc in question_type_acc.items():
            self.log(f"{q_type}_accuracy", acc, on_epoch=True, sync_dist=True, batch_size=self.batch_size)

        # Return predictions and metrics
        return {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "question_type_acc": question_type_acc,
            "predictions": pred,
            "answers": answer,
        }

        # todo:-------------------------
        # # Compute the test accuracy
        # test_acc = (pred.argmax(dim=1) == answer).float().mean()
        #
        # # Compute accuracy for each question type (e.g., yes/no)
        # question_type_acc = {}
        # for q_type in set(question_type):  # Iterate over unique question types
        #     indices = [i for i, q in enumerate(question_type) if q == q_type]
        #     if indices:  # If there are any examples for this question type
        #         q_type_preds = pred[indices].argmax(dim=1)
        #         q_type_answers = answer[indices]
        #         q_type_acc = (q_type_preds == q_type_answers).float().mean().item()
        #         question_type_acc[q_type] = round(q_type_acc * 100, 2)
        #
        # # Log the overall metrics
        # self.log("test_loss", test_loss, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        # self.log("test_acc", test_acc, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        #
        # # Log per-question-type accuracy (as numeric values, not formatted strings)
        # for q_type, acc in question_type_acc.items():
        #     self.log(f"test_acc_{q_type}", acc, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        #
        # # Log the formatted accuracy (for display purposes)
        # for q_type, acc in question_type_acc.items():
        #     self.log(f"{q_type}_accuracy", acc, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        #
        # # Return predictions and metrics
        # return {
        #     "test_loss": test_loss,
        #     "test_acc": test_acc,
        #     "question_type_acc": question_type_acc,
        #     "predictions": pred,
        #     "answers": answer,
        # }
    # todo:-------------------------
        return {
            "test_loss": test_loss,
        }

# 'Is it a rural or an urban area', 'Is there a grass area?', 'What is the number of roads?', 'Is there a road?', 'Is a large road present?', 'Are there less buildings than farmlands?', 'Is a residential building present?', 'Are there more commercial buildings than roads?', 'Is a forest present in the image?', 'What is the amount of farmlands?', 'How many grass areas are there?', 'What is the amount of residential buildings?', 'What is the number of circular commercial buildings in the image?', 'What is the number of buildings?', 'What is the number of water areas in the image?', 'Are there more residential buildings than water areas?', 'Are there more water areas than small commercial buildings?', 'Is a water area present?', 'Is a commercial building present in the image?', 'Are there less commercial buildings than water areas?', 'Is the number of residential buildings equal to the number of grass areas in the image?', 'Is a parking present?', 'Are there more small roads than residential buildings?', 'What is the number of rectangular commercial buildings?', 'Is there a medium building?', 'Is a rectangular road present?', 'Is the number of water areas equal to the number of residential buildings?', 'Is there a circular road?', 'Is there a building?', 'What is the amount of square buildings?', 'Are there more forests than residential buildings?', 'What is the number of orchards?', 'What is the amount of commercial buildings?', 'Is a square building present?', 'How many small residential buildings are there?', 'What is the amount of square water areas?', 'Is a small commercial building present?', 'How many small farmlands are there in the image?', 'Is a farmland next to a  commercial building present?', 'Are there less roads than buildings in the image?', 'Is there a small water area?', 'Is there a circular building?', 'Is there a farmland in the image?', 'Is there a circular water area?', 'Is a commercial building on the right of a  water area present?', 'Are there more roads than residential buildings?', 'Is a medium farmland present?', 'What is the number of circular roads?', 'Is the number of commercial buildings equal to the number of grass areas?', 'Is the number of roads equal to the number of commercial buildings?', 'How many forests are there?', 'How many pitchs are there?', 'What is the number of residential areas?', 'Are there more water areas than commercial buildings?', 'How many medium roads are there?', 'Is the number of grass areas equal to the number of water areas?', 'Is a circular residential building present?', 'Is there a square road?', 'How many small roads are there?', 'What is the amount of square grass areas?', 'Are there more rectangular residential buildings than place of worships?', 'Is there a medium water area?', 'Is the number of residential buildings equal to the number of water areas in the image?', 'Is there a large water area?']
# ['Is it a rural or an urban area', 'Is there a grass area?', 'What is the number of roads?', 'Is there a road?', 'Is a large road present?', 'Are there less buildings than farmlands?', 'Is a residential building present?', 'Are there more commercial buildings than roads?', 'Is a forest present in the image?', 'What is the amount of farmlands?', 'How many grass areas are there?', 'What is the amount of residential buildings?', 'What is the number of circular commercial buildings in the image?', 'What is the number of buildings?', 'What is the number of water areas in the image?', 'Are there more residential buildings than water areas?', 'Are there more water areas than small commercial buildings?', 'Is a water area present?', 'Is a commercial building present in the image?', 'Are there less commercial buildings than water areas?', 'Is the number of residential buildings equal to the number of grass areas in the image?', 'Is a parking present?', 'Are there more small roads than residential buildings?', 'What is the number of rectangular commercial buildings?', 'Is there a medium building?', 'Is a rectangular road present?', 'Is the number of water areas equal to the number of residential buildings?', 'Is there a circular road?', 'Is there a building?', 'What is the amount of square buildings?', 'Are there more forests than residential buildings?', 'What is the number of orchards?', 'What is the amount of commercial buildings?', 'Is a square building present?', 'How many small residential buildings are there?', 'What is the amount of square water areas?', 'Is a small commercial building present?', 'How many small farmlands are there in the image?', 'Is a farmland next to a  commercial building present?', 'Are there less roads than buildings in the image?', 'Is there a small water area?', 'Is there a circular building?', 'Is there a farmland in the image?', 'Is there a circular water area?', 'Is a commercial building on the right of a  water area present?', 'Are there more roads than residential buildings?', 'Is a medium farmland present?', 'What is the number of circular roads?', 'Is the number of commercial buildings equal to the number of grass areas?', 'Is the number of roads equal to the number of commercial buildings?', 'How many forests are there?', 'How many pitchs are there?', 'What is the number of residential areas?', 'Are there more water areas than commercial buildings?', 'How many medium roads are there?', 'Is the number of grass areas equal to the number of water areas?', 'Is a circular residential building present?', 'Is there a square road?', 'How many small roads are there?', 'What is the amount of square grass areas?', 'Are there more rectangular residential buildings than place of worships?', 'Is there a medium water area?', 'Is the number of residential buildings equal to the number of water areas in the image?', 'Is there a large water area?']