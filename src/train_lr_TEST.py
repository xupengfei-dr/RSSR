import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random
import copy
import torch

import Config_ALL
random.seed(42)
# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, [1]))
dir_path = './train_models_inner_3e5_dim192_inner_myencoder_plus_lr3_freezevision_fhead16168_encoder1812_large_o_in2pool_brifclassify_large_lr9_xiao_all_rssrblock__add+_adapter64'


torch.manual_seed(42)
import typer

import pytorch_lightning as pl
import torchvision.transforms as transforms

from augment.aug_lr import AutoAugment
from transformers import BertTokenizerFast, ViltImageProcessor, BlipImageProcessor, BlipProcessor, SiglipImageProcessor
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
    data_path = '/home/pengfei/DataSet/RSVQA_LR'
    LR_questionsJSON = os.path.join(data_path, 'LR_split_train_questions.json')
    LR_answersJSON = os.path.join(data_path, 'LR_split_train_answers.json')
    LR_imagesJSON = os.path.join(data_path, 'LR_split_train_images.json')
    LR_questionsvalJSON = os.path.join(data_path, 'LR_split_val_questions.json')
    LR_answersvalJSON = os.path.join(data_path, 'LR_split_val_answers.json')
    LR_imagesvalJSON = os.path.join(data_path, 'LR_split_val_images.json')
    LR_images_path = os.path.join(data_path, 'Images_LR/')
    LR_questionstestJSON = os.path.join(data_path, 'LR_split_test_questions.json')
    LR_answerstestJSON = os.path.join(data_path, 'LR_split_test_answers.json')
    LR_imagestestJSON = os.path.join(data_path, 'LR_split_test_images.json')
    # TODO:使用viltImageProcessor
    image_processor = BlipImageProcessor(do_resize=True, image_std=[0.229, 0.224, 0.225], image_mean=[0.485, 0.456, 0.406], do_rescale=True, do_normalize=True, size=256, size_divisor=32)
    processor = BlipProcessor.from_pretrained("/home/pengfei/blip-vqa-base")
    tokenizer = processor.tokenizer


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
                            selected_answers=LR_data_train.selected_answers,)


    LR_val_loader = torch.utils.data.DataLoader(LR_data_val, batch_size=batch_size, shuffle=False,
                                                num_workers=num_workers)


# todo:---------------------------------------------------------------------------------

    # loader for the validation data
    LR_data_test = VQALoader(LR_images_path,
                            LR_imagestestJSON,
                            LR_questionstestJSON,
                            LR_answerstestJSON,
                            tokenizer=tokenizer,
                            image_processor=image_processor,
                            Dataset='LR',
                            train=False,
                            ratio_images_to_use=1,
                            sequence_length=sequence_length,
                            selected_answers=LR_data_train.selected_answers, )

    LR_test_loader = torch.utils.data.DataLoader(LR_data_test, batch_size=batch_size, shuffle=False,
                                                num_workers=num_workers)



# todo:---------------------------------------------------------------------------------
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

    test_on_epoch_end_callback = EveryEpochTestCallback(LR_test_loader)


    trainer = pl.Trainer(devices=1,
                         accelerator='cuda',
                         fast_dev_run=False,
                         precision='16-mixed',
                         max_epochs=num_epochs,
                         num_sanity_val_steps=0,
                         #strategy='ddp_find_unused_parameters_true',
                         callbacks=[checkpoint_callback, early_stopping, lr_monitor,test_on_epoch_end_callback])

    trainer.fit(model, train_dataloaders=LR_train_loader, val_dataloaders=LR_val_loader, )

    # trainer.test(model,LR_val_loader)



class EveryEpochTestCallback(pl.Callback):
    """在每个训练周期结束后运行测试集的回调。"""
    def __init__(self, test_loader):
        super().__init__()
        self.test_loader = test_loader
    def on_train_epoch_end(self, trainer, pl_module):
        print("\n" + "=" * 50)
        print(f"  RUNNING TEST AT THE END OF TRAIN EPOCH {trainer.current_epoch}  ")
        print("=" * 50 + "\n")
        model_copy = copy.deepcopy(pl_module)
        temp_trainer = pl.Trainer(
            devices=trainer.device_ids,
            accelerator=trainer.accelerator,
            precision=trainer.precision,
            logger=False,
            callbacks=[],
            enable_progress_bar=True,
        )
        temp_trainer.test(model=model_copy, dataloaders=self.test_loader, verbose=True)


if __name__ == "__main__":
    Config_ALL.TRAIN_HR=False
    typer.run(main)
