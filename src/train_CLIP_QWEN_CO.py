import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import random
import sys

import torch

import Config_ALL

random.seed(42)
# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, [1]))

dir_path = './train_lr_clip_qwen_coattention'

sys.path.append("../")
sys.path.append("../../")

torch.manual_seed(42)
import typer

import pytorch_lightning as pl
import torchvision.transforms as transforms

from augment.aug_lr import AutoAugment
from transformers import BertTokenizerFast, ViltImageProcessor, BlipImageProcessor, BlipProcessor, AutoTokenizer, \
    AutoProcessor
# from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from model.RSVQA_model_test import VQAModel
from dataloader.VQALoader_TestLR import VQALoader


def main(num_workers: int = 12,
         ratio_images_to_use: float = 1,
         sequence_length: int = 40,
         num_epochs: int = 70,
         batch_size: int = 64,
         lr: float = 3e-5,
         Dataset='LR'):
    data_path = '/home/pengfei/rsvqa/data/RSVQA_LR'
    LR_questionsJSON = os.path.join(data_path, 'Image/LR_split_train_questions.json')
    LR_answersJSON = os.path.join(data_path, 'Image/LR_split_train_answers.json')
    LR_imagesJSON = os.path.join(data_path, 'Image/LR_split_train_images.json')
    LR_questionsvalJSON = os.path.join(data_path, 'Image/LR_split_val_questions.json')
    LR_answersvalJSON = os.path.join(data_path, 'Image/LR_split_val_answers.json')
    LR_imagesvalJSON = os.path.join(data_path, 'Image/LR_split_val_images.json')
    LR_images_path = os.path.join(data_path, 'Image/Images_LR/')

    qwen_model_path = "/home/pengfei/Qwen2.5-1.5B-Instruct"
    clip_vit_model_Path = "/home/pengfei/clip-vit-large-patch14-336"

    tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)
    image_processor = AutoProcessor.from_pretrained(clip_vit_model_Path, size=256, )

    # TODO:使用viltImageProcessor
    # 加载 BLIP 的 tokenizer

    # image_processor = BlipImageProcessor(do_resize=True, image_std=[0.229, 0.224, 0.225], image_mean=[0.485, 0.456, 0.406], do_rescale=True, do_normalize=True, size=256, size_divisor=32)
    # 使用相同的 processor 和 tokenizer
    # processor = BlipProcessor.from_pretrained("/home/pengfei/blip-vqa-capfilt-large")
    # tokenizer = processor.tokenizer

    if Dataset == 'LR':
        model = VQAModel(batch_size=batch_size, lr=lr, number_outputs=9)
    else:
        model = VQAModel(batch_size=batch_size, lr=lr, number_outputs=98)

    transform_train = [
        transforms.RandomHorizontalFlip(),
    ]
    transform_train.append(AutoAugment())
    transform_train = transforms.Compose(transform_train)

    # loader for the training data
    LR_data_train = VQALoader(LR_images_path,
                              LR_imagesJSON,
                              LR_questionsJSON,
                              LR_answersJSON,
                              tokenizer=tokenizer,
                              image_processor=image_processor,
                              Dataset='LR',
                              train=True,
                              sequence_length=sequence_length,
                              ratio_images_to_use=ratio_images_to_use,
                              transform=transform_train,
                              )

    LR_train_loader = torch.utils.data.DataLoader(LR_data_train, batch_size=batch_size, shuffle=True,
                                                  num_workers=num_workers)

    # loader for the validation data
    LR_data_val = VQALoader(LR_images_path,
                            LR_imagesvalJSON,
                            LR_questionsvalJSON,
                            LR_answersvalJSON,
                            tokenizer=tokenizer,
                            image_processor=image_processor,
                            Dataset='LR',
                            train=False,
                            ratio_images_to_use=1,
                            sequence_length=sequence_length,
                            selected_answers=LR_data_train.selected_answers, )

    LR_val_loader = torch.utils.data.DataLoader(LR_data_val, batch_size=batch_size, shuffle=False,
                                                num_workers=num_workers)

    # wandb_logger = WandbLogger(project='RSVQA_LR')

    # specify how to checkpoint
    checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                          monitor="valid_acc",
                                          save_weights_only=True,
                                          mode="max",
                                          dirpath=dir_path,
                                          filename=f"{{epoch}}_{{valid_acc:.5f}}_{{acc_count:.5f}}_{{acc_rural_urban:.5f}}_{{acc_presence:.5f}}")

    # early stopping
    early_stopping = EarlyStopping(monitor="valid_acc", patience=20, mode="max")

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(devices=1,
                         accelerator='cuda',
                         fast_dev_run=False,
                         precision='16-mixed',
                         max_epochs=num_epochs,
                         num_sanity_val_steps=0,
                         # strategy='ddp_find_unused_parameters_true',
                         callbacks=[checkpoint_callback, early_stopping, lr_monitor])

    trainer.fit(model, train_dataloaders=LR_train_loader, val_dataloaders=LR_val_loader, )

    # trainer.test(model,LR_val_loader)


if __name__ == "__main__":
    Config_ALL.TRAIN_HR = False
    typer.run(main)
