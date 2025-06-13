import os
# os.environ['CUDA_VISIBLE_DEVICES']="1"
import argparse

import math
import pandas as pd
import numpy as np 

import torch
import pdb
from nets.segmentation import FasterTTRCNN,TTRCNN
from loaders.tt_dataset import TTDataModuleBX, TrainTransformsSeg, EvalTransformsSeg,BBXImageTrainTransform, BBXImageEvalTransform, BBXImageTestTransform
from callbacks.logger import SegImageLoggerNeptune, MaskRCNNImageLoggerNeptune,FasterRCNNImageLoggerNeptune
from  torchvision import models
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.loggers import NeptuneLogger, TensorBoardLogger

from sklearn.utils import class_weight

def remove_labels(df, args):

    # df = df.drop_duplicates(subset=['x_patch', 'y_patch', 'filename'])
    df = df.loc[ df['to_drop'] == 0]

    if args.drop_labels is not None:
        df = df[ ~ df[args.label_column].isin(args.drop_labels)]

    if args.concat_labels is not None:
        replacement_val = df.loc[ df[args.label_column] == args.concat_labels[0]][args.class_column].unique()
        df.loc[ df[args.label_column].isin(args.concat_labels), args.class_column ] = replacement_val[0]

    unique_classes = sorted(df[args.class_column].unique())
    class_mapping = {value: idx+1 for idx, value in enumerate(unique_classes)}

    df[args.class_column] = df[args.class_column].map(class_mapping)    
    # df.loc[ df[args.label_column] == 'Reject', class_column]  = 0

    print(f"Kept Classes : {df[args.label_column].unique()}, {class_mapping}")

    return df


def main(args):


    df_train = pd.read_csv(args.csv_train)
    df_val = pd.read_csv(args.csv_valid)
    df_test = pd.read_csv(args.csv_test)

    df_train = remove_labels(df_train, args)
    df_val = remove_labels(df_val, args)
    df_test = remove_labels(df_test, args)

    args_params = vars(args)
    unique_classes = np.sort(np.unique(df_train[args.class_column]))
    args_params['out_features'] = len(unique_classes) + 1

    args_params['class_weights'] = np.ones(args_params['out_features'])
    if args.balanced_weights:
        unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[args.class_column]))
        args_params['class_weights'] =  np.concatenate((np.array([0]), unique_class_weights))

    elif args.custom_weights:
        args_params['class_weights'] = np.array(args.custom_weights)
    
    elif args.balanced:        

        g_train = df_train.groupby(args.class_column)
        df_train = g_train.apply(lambda x: x.sample(g_train.size().min())).reset_index(drop=True).sample(frac=1).reset_index(drop=True)

        unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y= df_train[args.class_column]))
        g_val = df_val.groupby(args.class_column)
        df_val = g_val.apply(lambda x: x.sample(g_val.size().min())).reset_index(drop=True).sample(frac=1).reset_index(drop=True)
        args_params['class_weights'] =  np.concatenate((np.array([0]), unique_class_weights)) 

        print(f"{df_train[[args.label_column, args.class_column]].value_counts()}")

    print(f"class weights: {args_params['class_weights']}")

    ttdata = TTDataModuleBX(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column=args.img_column, class_column=args.class_column,
                            mount_point=args.mount_point, train_transform=BBXImageTrainTransform(), valid_transform=BBXImageEvalTransform(), test_transform=BBXImageTestTransform())


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss',
        save_last=True,
    )

    image_logger = FasterRCNNImageLoggerNeptune(log_steps = args.log_every_n_steps)
    if args.model:
        model = FasterTTRCNN.load_from_checkpoint(args.model, **vars(args), strict=False)
    else:
        model = FasterTTRCNN(**vars(args))
    
    
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="max")
    logger = None
    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)
    elif args.neptune_tags:
        logger = NeptuneLogger(project='ImageMindAnalytics/trachoma',
                               tags=args.neptune_tags,
                               api_key=os.environ['NEPTUNE_API_TOKEN'],
                               log_model_checkpoints=False)

    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback, checkpoint_callback, image_logger],
        devices=torch.cuda.device_count(), 
        accelerator="gpu", 
        strategy=DDPStrategy(find_unused_parameters=False),
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=.5,
    )
    trainer.fit(model, datamodule=ttdata, ckpt_path=args.model)

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='TT object detection Training')

    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--model', help='Model to continue training', type=str, default= None)
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--csv_train', required=True, type=str, help='Train CSV')
    input_group.add_argument('--csv_valid', required=True, type=str, help='Valid CSV')
    input_group.add_argument('--csv_test', required=True, type=str, help='Test CSV')
    input_group.add_argument('--img_column', type=str, default="img_path", help='Name of image column in csv')
    input_group.add_argument('--class_column', type=str, default="class", help='Name of segmentation column in csv')
    input_group.add_argument('--label_column', type=str, default="label", help='Name of label column in csv')
    input_group.add_argument('--drop_labels', type=str, default=None, nargs='+', help='drop labels in dataframe')
    input_group.add_argument('--concat_labels', type=str, default=None, nargs='+', help='concat labels in dataframe')

    weight_group = input_group.add_mutually_exclusive_group()
    weight_group.add_argument('--balanced_weights', type=int, default=0, help='Compute weights for balancing the data')
    weight_group.add_argument('--custom_weights', type=float, default=None, nargs='+', help='Custom weights for balancing the data')
    weight_group.add_argument('--balanced', type=int, default=None, help='balance dataframe')

    hparams_group = parser.add_argument_group('Hyperparameters')
    hparams_group.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    hparams_group.add_argument('--wd', '--weight_decay', default=0.001, type=float, help='weight decay for AdamW')
    hparams_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    hparams_group.add_argument('--patience', help='Max number of patience steps for EarlyStopping', type=int, default=30)
    hparams_group.add_argument('--steps', help='Max number of steps per epoch', type=int, default=-1)    
    hparams_group.add_argument('--batch_size', help='Batch size', type=int, default=256)
    # hparams_group.add_argument('--loss_type', help='choice of loss', type=str, default='cross-entropy', choices=['cross-entropy', 'focal'])
    hparams_group.add_argument('--loss_weights', help='custom loss weights [classifier, box_reg, objectness (rpn), box_reg (rpn)]', type=float, nargs='+', default=[1.0, 1.0, 1.0, 1.0])

    
    logger_group = parser.add_argument_group('Logger')
    logger_group.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=1)
    logger_group.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    logger_group.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="segmentation_unet")
    
    logger_group.add_argument('--neptune_tags', help='neptune tag', type=str, nargs='+', default=None)

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output', type=str, default="./")

    args = parser.parse_args()

    main(args)
