import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random
import torch
import Config_ALL

random.seed(42)
torch.manual_seed(42)
import typer
import pytorch_lightning as pl
import torchvision.transforms as transforms

from augment.aug_lr import AutoAugment
from transformers import BertTokenizerFast, ViltImageProcessor, BlipImageProcessor, BlipProcessor
from pytorch_lightning.loggers import WandbLogger
from model.RSVQA_model_test import VQAModel as VQAModel2

from dataloader.VQALoader_TestLR import VQALoader

# path = '/home/pengfei/mamba/src/train_models_inner_3e5_dim192_inner_myencoder_plus_lr3_ freezevision_fhead16168_layer18_large/epoch=39_valid_acc=0.89683_acc_count=0.78494_acc_rural_urban=0.92000_acc_presence=0.93338.ckpt'
# path = '/home/pengfei/mamba/src/train_models_inner_3e5_dim192_inner_myencoder_plus_lr3_freezevision_fhead16168_encoder18_large_o/epoch=48_valid_acc=0.89714_acc_count=0.78697_acc_rural_urban=0.90000_acc_presence=0.93235.ckpt'
# path = '/home/pengfei/mamba/src/train_models_inner_3e5_dim192_inner_myencoder_plus_lr3_freezevision_fhead16168_encoder18_large_o/epoch=47_valid_acc=0.89453_acc_count=0.78393_acc_rural_urban=0.93000_acc_presence=0.92960.ckpt'
path = '/home/gpuadmin/RSSR/src/train_models_inner_3e5_dim192_inner_myencoder_plus_lr3_freezevision_fhead16168_encoder1812_large_o_in2pool_brifclassify_large_lr9_xiao_all_rssrblock__add+/epoch=12_valid_acc=0.89275_acc_count=0.78287_acc_rural_urban=0.91000_acc_presence=0.92703.ckpt'

def main(num_workers: int = 12,
         ratio_images_to_use: float = 1,
         sequence_length: int = 40,
         num_epochs: int = 70,
         batch_size: int = 64,
         lr: float = 1e-3,
         Dataset='LR'):
    data_path = '/home/gpuadmin/DataSet/RSVQA_LR'
    LR_questionstestJSON = os.path.join(data_path, 'LR_split_test_questions.json')
    LR_answerstestJSON = os.path.join(data_path, 'LR_split_test_answers.json')
    LR_imagestestJSON = os.path.join(data_path, 'LR_split_test_images.json')
    LR_images_path = os.path.join(data_path, 'Images_LR/')
    # LR_questionstestJSON = os.path.join(data_path, 'LR_split_val_questions.json')
    # LR_answerstestJSON = os.path.join(data_path, 'LR_split_val_answers.json')
    # LR_imagestestJSON = os.path.join(data_path, 'LR_split_val_images.json')
    # LR_images_path = os.path.join(data_path, 'Images_LR/')

    # TODO:使用viltImageProcessor
    # 加载 BLIP 的 tokenizer
    image_processor = BlipImageProcessor(padding=True, do_resize=True, image_std=[0.229, 0.224, 0.225],
                                         image_mean=[0.485, 0.456, 0.406], do_rescale=True, do_normalize=True, size=256,
                                         size_divisor=32)
    # 使用相同的 processor 和 tokenizer
    processor = BlipProcessor.from_pretrained("/home/gpuadmin/blip-vqa-base")
    tokenizer = processor.tokenizer


    transform_train = [
        transforms.RandomHorizontalFlip(),
    ]
    transform_train.append(AutoAugment())
    transform_train = transforms.Compose(transform_train)

    # loader for the test data
    LR_data_test = VQALoader(LR_images_path,
                             LR_imagestestJSON,
                             LR_questionstestJSON,
                             LR_answerstestJSON,
                             train=False,
                             tokenizer=tokenizer,
                             image_processor=image_processor,
                             Dataset='LR',
                             sequence_length=sequence_length,
                             ratio_images_to_use=ratio_images_to_use,
                             transform=transform_train, )

    LR_test_loader = torch.utils.data.DataLoader(LR_data_test, batch_size=batch_size,shuffle=False,
                                                 num_workers=num_workers,drop_last=True)


    trainer = pl.Trainer(
        devices=1,
        accelerator='cuda',
        # logger=wandb_logger,
        num_sanity_val_steps=0,
        fast_dev_run=False,

    )


    print("-----------------------------------------")
    checkpoint_path = path
    # 加载 .ckpt 文件
    def print_all_keys(d,model):
        """
        循环遍历并打印字典中的所有键。
        :param d: 输入的字典（通常是 state_dict）
        """
        # 检查字典是否是最外层的字典
        if isinstance(d, dict):
            for k, v in d.items():
                print(f'Key: {k}')  # 打印当前的键
                setattr(model, k, v)

    def check_all_keys(d,model):
        """
        循环遍历并打印字典中的所有键。
        :param d: 输入的字典（通常是 state_dict）
        """
        # 检查字典是否是最外层的字典
        if isinstance(d, dict):
            for k, v in d.items():
                vm = getattr(model, k)
                if torch.equal(vm,v):
                    print("ok")
                else:
                    print("no")

    # 加载 checkpoint 文件
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']
    model2 = VQAModel2(64, 2e-4, number_outputs=9)
    model2.load_state_dict(checkpoint['state_dict'], strict=False)

    # 打印所有的键
    print_all_keys(state_dict,model2)
    check_all_keys(state_dict,model2)
    model2.eval()

    print(model2)


    print("-----------------------")

    select_answer = ['yes', 'no', 'between 0 and 10', '0', 'between 10 and 100', 'between 100 and 1000', 'more than 1000', 'rural', 'urban']

    print("Starting testing...")
    trainer.test(model2, dataloaders=LR_test_loader)

if __name__ == "__main__":
    Config_ALL.TRAIN_HR = False
    typer.run(main)

