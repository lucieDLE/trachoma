import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch

from nets import segmentation
from loaders.tt_dataset import TTDataModuleSeg, TrainTransformsFullSeg, EvalTransformsFullSeg

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

from pytorch_lightning.callbacks import QuantizationAwareTraining
from torch.utils.mobile_optimizer import optimize_for_mobile

from sklearn.utils import class_weight

from callbacks.logger import SegYOLOImageLogger


def replace_last(str, old, new):
    if old not in str:
        return str
    idx = str.rfind(old)
    return str[:idx] + new + str[idx+len(old):]

def main(args):

    if(os.path.splitext(args.csv_train)[1] == ".csv"):
        df_train = pd.read_csv(args.csv_train) 
        df_val = pd.read_csv(args.csv_valid)
        df_test = pd.read_csv(args.csv_test)
    else:
        df_train = pd.read_parquet(args.csv_train)
        df_val = pd.read_parquet(args.csv_valid)
        df_test = pd.read_parquet(args.csv_test)
    
    NN = getattr(segmentation, args.nn)

    args_params = vars(args)    

    model = NN(**args_params)

    train_transform = TrainTransformsFullSeg()
    eval_transform = EvalTransformsFullSeg()

    ttdata = TTDataModuleSeg(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column=args.img_column, seg_column=args.seg_column, mount_point=args.mount_point, train_transform=train_transform, valid_transform=eval_transform, test_transform=eval_transform)


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )

    image_logger = SegYOLOImageLogger(log_steps=args.log_every_n_steps, num_images=2)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")

    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)    

    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback, checkpoint_callback, image_logger],
        devices=torch.cuda.device_count(), 
        accelerator="gpu", 
        strategy=DDPStrategy(find_unused_parameters=False),
        log_every_n_steps=args.log_every_n_steps,
        accumulate_grad_batches=args.accumulate_grad_batches
    )
    trainer.fit(model, datamodule=ttdata, ckpt_path=args.model)
    


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Surgery prediction Training')

    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--model', help='Model to continue training', type=str, default= None)
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--csv_train', required=True, type=str, help='Train CSV')
    input_group.add_argument('--csv_valid', required=True, type=str, help='Valid CSV')
    input_group.add_argument('--csv_test', required=True, type=str, help='Test CSV')
    input_group.add_argument('--img_column', type=str, default="img_path", help='Name of image column in csv')
    input_group.add_argument('--seg_column', type=str, default="seg_path", help='Name of segmentation column in csv')    

    hparams_group = parser.add_argument_group('Hyperparameters')
    
    hparams_group.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')    
    hparams_group.add_argument('--accumulate_grad_batches', default=1, type=int, help='Accumulate gradient steps')
    hparams_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    hparams_group.add_argument('--patience', help='Max number of patience steps for EarlyStopping', type=int, default=30)
    hparams_group.add_argument('--steps', help='Max number of steps per epoch', type=int, default=-1)    
    hparams_group.add_argument('--batch_size', help='Batch size', type=int, default=256)
    hparams_group.add_argument('--nn', help='Type of PL neural network', type=str, default="MobileYOLO")
    hparams_group.add_argument('--base_encoder', help='Type of torchvision neural network', type=str, default="efficientnet_b0")        
    hparams_group.add_argument('--size_bb', help='Size of the bounding box for the bounding box prediction task', type=int, default=448)
    hparams_group.add_argument('--pad', help='Pad the bounding box', type=float, default=0.1)
    
    
    logger_group = parser.add_argument_group('Logger')
    logger_group.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=10)
    logger_group.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    logger_group.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="segmentation_unet")
    

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output', type=str, default="./")

    args = parser.parse_args()

    main(args)
