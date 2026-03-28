import importlib
import os
import random
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 可选 'high', 'medium', 'low'

sys.path.append("../")
TRAIN_HR =False
import torch
torch.set_float32_matmul_precision('medium')

random.seed(42)
# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, [1]))
# torch.cuda.set_device(7)
path = './train_models_hr_4e6_batch256_rssr12_vqalarge_freeze_all_vision_fhead16168_encoder1612_128_e15_inpool_brify_lr3_gelu_size384_all_xiao__rssradapter'
torch.manual_seed(42)
import typer

import pytorch_lightning as pl
import torchvision.transforms as transforms
from augment.auto_augment import AutoAugment
from transformers import BertTokenizerFast, ViltImageProcessor, BlipImageProcessor, BlipProcessor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from model.RSVQA_model_test import VQAModel
from dataloader.VQALoader_HR import VQALoader


def main(num_workers: int = 12,
         ratio_images_to_use: int = 1,
         sequence_length: int = 40,
         num_epochs: int = 12,
         batch_size: int = 64,
         lr: float = 4e-6,
         Dataset='HR'):
    data_path = '/home/gpuadmin/DataSet/RSVQA_HR'

    HR_questionsJSON = os.path.join(data_path, 'USGS_split_train_questions.json')
    HR_answersJSON = os.path.join(data_path, 'USGS_split_train_answers.json')
    HR_imagesJSON = os.path.join(data_path, 'USGS_split_train_images.json')



    HR_questionsvalJSON = os.path.join(data_path, 'USGS_split_val_questions.json')
    HR_answersvalJSON = os.path.join(data_path, 'USGS_split_val_answers.json')
    HR_imagesvalJSON = os.path.join(data_path, 'USGS_split_val_images.json')

    HR_questionstestJSON = os.path.join(data_path, 'USGS_split_test_questions.json')
    HR_answerstestJSON = os.path.join(data_path, 'USGS_split_test_answers.json')
    HR_imagestestJSON = os.path.join(data_path, 'USGS_split_test_images.json')
    HR_images_path = os.path.join(data_path, 'Data/')

    # HR_questionstestJSON = os.path.join(data_path, 'USGS/USGS_split_test_phili_questions.json')
    # HR_answerstestJSON = os.path.join(data_path, 'USGS/USGS_split_test_phili_answers.json')
    # HR_imagestestJSON = os.path.join(data_path, 'USGS/USGS_split_test_phili_images.json')

    # tokenizer = BertTokenizerFast.from_pretrained('dandelin/vilt-b32-mlm')
    # image_processor = ViltImageProcessor(do_resize=True, image_std=[0.229, 0.224, 0.225],
    #                                      image_mean=[0.485, 0.456, 0.406], do_rescale=True, do_normalize=True, size=512,
    #                                      size_divisor=32)
    # TODO:使用viltImageProcessor
    # 加载 BLIP 的 tokenizer
    image_processor = BlipImageProcessor(padding=True, do_resize=True, image_std=[0.229, 0.224, 0.225],
                                         image_mean=[0.485, 0.456, 0.406], do_rescale=True, do_normalize=True, size=384,
                                         size_divisor=32)
    # 使用相同的 processor 和 tokenizer
    # processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    processor = BlipProcessor.from_pretrained("/home/gpuadmin/blip-vqa-base")

    tokenizer = processor.tokenizer

    if Dataset == 'LR':
        model = VQAModel(batch_size=batch_size, lr=lr, number_outputs=9)
    else:
        model = VQAModel(batch_size=batch_size, lr=lr, number_outputs=94)

    transform_train = [
        transforms.RandomHorizontalFlip(),
    ]
    transform_train.append(AutoAugment())
    transform_train = transforms.Compose(transform_train)
    # loader for the training data
    HR_data_train = VQALoader(HR_images_path,
                              HR_imagesJSON,
                              HR_questionsJSON,
                              HR_answersJSON,
                              tokenizer=tokenizer,
                              image_processor=image_processor,
                              Dataset='HR',
                              train=True,
                              sequence_length=sequence_length,
                              ratio_images_to_use=ratio_images_to_use,
                              transform=transform_train)

    HR_train_loader = torch.utils.data.DataLoader(HR_data_train, batch_size=batch_size, shuffle=True,
                                                  num_workers=num_workers)

    # loader for the validation data
    HR_data_val = VQALoader(HR_images_path,
                            HR_imagesvalJSON,
                            HR_questionsvalJSON,
                            HR_answersvalJSON,
                            tokenizer=tokenizer,
                            image_processor=image_processor,
                            Dataset='HR',
                            train=False,
                            ratio_images_to_use=ratio_images_to_use,
                            sequence_length=sequence_length, )

    HR_val_loader = torch.utils.data.DataLoader(HR_data_val, batch_size=batch_size, shuffle=False,
                                                num_workers=num_workers)

    # wandb_logger = WandbLogger(project='RSVQA_HR')

    # specify how to checkpoint
    checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                          monitor="valid_acc",
                                          save_weights_only=True,
                                          mode="max",
                                          dirpath=path,
                                          filename=f"{{epoch}}_{{valid_acc:.5f}}_{{acc_count:.5f}}_{{acc_rural_urban:.5f}}_{{acc_presence:.5f}}")

    # early stopping
    early_stopping = EarlyStopping(monitor="valid_acc", patience=10, mode="max")

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(devices=1,
                         accelerator='cuda',
                         fast_dev_run=False,
                         precision='16-mixed',
                         max_epochs=num_epochs,
                         # logger=wandb_logger,
                         # strategy='ddp_find_unused_parameters_true',
                         num_sanity_val_steps=0,
                         callbacks=[checkpoint_callback, early_stopping, lr_monitor])

    trainer.fit(model, train_dataloaders=HR_train_loader, val_dataloaders=HR_val_loader)


if __name__ == "__main__":
    TRAIN_HR = True  # 如果要使用256*256的图像作为输入，则设置为False ，若512*512则设置为true
    typer.run(main)
