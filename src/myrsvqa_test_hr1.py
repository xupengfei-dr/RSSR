import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import random
import sys

import Config_ALL

sys.path.append("../")

import torch

random.seed(42)
# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, [1]))
# torch.cuda.set_device(7)
# test_path = '/home/pengfei/mamba/src/train_models_hr1_2e6_256_batch64_rssr12_vqalarge_freezevision_fhead16168_encoder18_256/epoch=4_valid_acc=0.84286_acc_count=0.67008_acc_rural_urban=0.85441_acc_presence=0.92453.ckpt'
# test_path = '/home/pengfei/mamba/src/train_models_hr_2e6_256_batch256_rssr12_vqabase_freezedecoder_fhead16168_encoder18_64_e20/epoch=3_valid_acc=0.85317_acc_count=0.68400_acc_rural_urban=0.86566_acc_presence=0.93096.ckpt'
# test_path = '/home/pengfei/mamba/src/train_models_hr_2e6_256_batch256_rssr12_vqabase_freezedecoder_fhead16168_encoder18_64_e20/epoch=4_valid_acc=0.85396_acc_count=0.68698_acc_rural_urban=0.86454_acc_presence=0.93121.ckpt'
test_path = '/home/pengfei/mamba/src/train_models_hr_4e6_batch256_rssr12_vqalarge_freeze_all_vision_fhead16168_encoder1612_128_e15_inpool_brify_lr3_gelu_size384_all_xiao_rssradapter_rssrblock/epoch=8_valid_acc=0.78741_acc_count=0.65737_acc_rural_urban=0.83959_acc_presence=0.90621.ckpt'
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


def main(num_workers: int = 16,
         ratio_images_to_use: int = 1,
         sequence_length: int = 40,
         num_epochs: int = 20,
         batch_size: int = 64,
         lr: float = 1e-4,
         Dataset='HR'):
    data_path = '/home/pengfei/RSVQA_HR'

    HR_questionsJSON = os.path.join(data_path, 'USGS_split_train_questions.json')
    HR_answersJSON = os.path.join(data_path, 'USGS_split_train_answers.json')
    HR_imagesJSON = os.path.join(data_path, 'USGS_split_train_images.json')
    HR_questionsvalJSON = os.path.join(data_path, 'USGS_split_val_questions.json')
    HR_answersvalJSON = os.path.join(data_path, 'USGS_split_val_answers.json')
    HR_imagesvalJSON = os.path.join(data_path, 'USGS_split_val_images.json')

    # todo : 1
    # HR_questionstestJSON = os.path.join(data_path, 'USGS_split_test_questions.json')
    # HR_answerstestJSON = os.path.join(data_path, 'USGS_split_test_answers.json')
    # HR_imagestestJSON = os.path.join(data_path, 'USGS_split_test_images.json')
    HR_images_path = os.path.join(data_path, 'Data/')


# todo : 2
    HR_questionstestJSON = os.path.join(data_path, 'USGS_split_test_phili_questions.json')
    HR_answerstestJSON = os.path.join(data_path, 'USGS_split_test_phili_answers.json')
    HR_imagestestJSON = os.path.join(data_path, 'USGS_split_test_phili_images.json')

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
    processor = BlipProcessor.from_pretrained("/home/pengfei/blip-vqa-capfilt-large")
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
    HR_data_test = VQALoader(HR_images_path,
                             HR_imagestestJSON,
                             HR_questionstestJSON,
                             HR_answerstestJSON,
                             tokenizer=tokenizer,
                             image_processor=image_processor,
                             Dataset='HR',
                             train=False,
                             ratio_images_to_use=ratio_images_to_use,
                             transform=transform_train,
                             sequence_length=sequence_length, )

    HR_test_loader = torch.utils.data.DataLoader(HR_data_test, batch_size=batch_size, shuffle=False,
                                                 num_workers=num_workers)

    # wandb_logger = WandbLogger(project='RSVQA_HR')

    # specify how to checkpoint
    # checkpoint_callback = ModelCheckpoint(save_top_k=1,
    #                                       monitor="valid_acc",
    #                                       save_weights_only=True,
    #                                       mode="max",
    #                                       dirpath=path,
    #                                       filename=f"{{epoch}}_{{valid_acc:.5f}}_{{acc_count:.5f}}_{{acc_rural_urban:.5f}}_{{acc_presence:.5f}}")

    # early stopping
    # early_stopping = EarlyStopping(monitor="valid_acc", patience=10, mode="max")
    # lr_monitor = LearningRateMonitor(logging_interval='step')

    # trainer = pl.Trainer(devices=1,
    #                      accelerator='cuda',
    #                      fast_dev_run=False,
    #                      precision='16-mixed',
    #                      max_epochs=num_epochs,
    #                      logger=wandb_logger,
    #                      # strategy='ddp_find_unused_parameters_true',
    #                      num_sanity_val_steps=0,
    #                      callbacks=[checkpoint_callback, early_stopping, lr_monitor])
    trainer = pl.Trainer(
        devices=1,  # 使用 1 个 GPU
        accelerator='cuda',  # 使用 GPU 加速
        # logger=wandb_logger,  # 你可以继续使用 wandb 进行日志记录
        num_sanity_val_steps=0,  # 跳过 sanity check，只进行测试
        fast_dev_run=False,  # 不做快速开发运行（因为我们只关心测试）

    )
    # trainer.fit(model, train_dataloaders=HR_train_loader, val_dataloaders=HR_val_loader)

    print("-----------------------------------------")
    checkpoint_path = test_path

    # 加载 .ckpt 文件
    def print_all_keys(d, model):
        """
        循环遍历并打印字典中的所有键。
        :param d: 输入的字典（通常是 state_dict）
        """
        # 检查字典是否是最外层的字典
        if isinstance(d, dict):
            for k, v in d.items():
                print(f'Key: {k}')  # 打印当前的键
                setattr(model, k, v)

    def check_all_keys(d, model):
        """
        循环遍历并打印字典中的所有键。
        :param d: 输入的字典（通常是 state_dict）
        """
        # 检查字典是否是最外层的字典
        if isinstance(d, dict):
            for k, v in d.items():
                vm = getattr(model, k)
                if torch.equal(vm, v):
                    print("ok")
                else:
                    print("no")

    # 加载 checkpoint 文件
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']
    model2 = VQAModel(64, 2e-4, number_outputs=94)
    model2.load_state_dict(checkpoint['state_dict'], strict=False)
    # 打印所有的键
    print_all_keys(state_dict, model2)
    check_all_keys(state_dict, model2)
    model2.eval()
    trainer.test(model2, HR_test_loader)


if __name__ == "__main__":
    Config_ALL.TRAIN_HR = False  # 如果要使用256*256的图像作为输入，则设置为False ，若512*512则设置为true
    typer.run(main)
