import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch

from nets.classification import EfficientnetV2s, EfficientnetV2sStacks, EfficientnetV2sStacksDot, EfficientnetV2sStacksSigDot, MobileNetV2, MobileNetV2Stacks
from loaders.tt_dataset import TTDataModuleStacks

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

from sklearn.utils import class_weight

def main(args):

    # train_fn = os.path.join(args.mount_point, 'Analysis_Set_20220422', 'trachoma_bsl_mtss_besrat_field_patches_train_20220422_fold4_train.csv')
    # valid_fn = os.path.join(args.mount_point, 'Analysis_Set_20220422', 'trachoma_bsl_mtss_besrat_field_patches_train_20220422_fold4_test.csv')
    # test_fn = os.path.join(args.mount_point, 'Analysis_Set_20220422', 'trachoma_bsl_mtss_besrat_field_patches_test_20220422.csv')

    # train_fn = os.path.join(args.mount_point, 'Analysis_Set_20220422', 'trachoma_bsl_mtss_besrat_field_train_20220422_stacks_train.csv')
    # valid_fn = os.path.join(args.mount_point, 'Analysis_Set_20220422', 'trachoma_bsl_mtss_besrat_field_train_20220422_stacks_eval.csv')
    # test_fn = os.path.join(args.mount_point, 'Analysis_Set_20220422', 'trachoma_bsl_mtss_besrat_field_test_20220422_stacks.csv')

    train_fn = os.path.join(args.mount_point, 'Analysis_Set_202208', 'trachoma_bsl_mtss_besrat_field_train_202208_stacks_train.csv')
    valid_fn = os.path.join(args.mount_point, 'Analysis_Set_202208', 'trachoma_bsl_mtss_besrat_field_train_202208_stacks_eval.csv')
    test_fn = os.path.join(args.mount_point, 'Analysis_Set_202208', 'trachoma_bsl_mtss_besrat_field_train_202208_stacks_eval.csv')

    class_column = "class"
    img_column = "image"
    df_train = pd.read_csv(train_fn)    

    unique_classes = np.sort(np.unique(df_train[class_column]))
    unique_class_weights = np.array(class_weight.compute_class_weight(class_weight={0: 1, 1: 8}, classes=unique_classes, y=df_train[class_column]))

    class_replace = {}
    for cn, cl in enumerate(unique_classes):
        class_replace[cl] = cn
    print(unique_classes, unique_class_weights, class_replace)

    df_train[class_column] = df_train[class_column].replace(class_replace).astype(int)

    df_val = pd.read_csv(valid_fn)        
    df_val[class_column] = df_val[class_column].replace(class_replace).astype(int)
    
    df_test = pd.read_csv(test_fn)
    df_test[class_column] = df_test[class_column].replace(class_replace).astype(int)
    
    ttdata = TTDataModuleStacks(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column=img_column, class_column=class_column)


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )

    model_patches = None
    if args.model_patches and "efficientnet_v2s" in args.nn:
        model_patches = EfficientnetV2s(args, out_features=3, features=True).load_from_checkpoint(args.model_patches)        
        model_patches.eval()
    elif args.model_patches and args.nn == "mobilenet_v2_stacks":
        model_patches = MobileNetV2(args, out_features=3, features=True).load_from_checkpoint(args.model_patches)
        model_patches.eval()
    
    
    if args.nn == "efficientnet_v2s_stacks":
        model = EfficientnetV2sStacks(args, out_features=unique_classes.shape[0], class_weights=unique_class_weights, model_patches=model_patches)
    elif args.nn == "efficientnet_v2s_stacks_dot":
        model = EfficientnetV2sStacksDot(args, out_features=unique_classes.shape[0], class_weights=unique_class_weights, model_patches=model_patches)

        print("START LOADING!")
        model.F.load_state_dict(torch.load('/work/jprieto/data/trachoma/train/classification/Analysis_Set_202208/stacks_v9_efficientnet_v2s_dot_recall_w8/epoch=26-val_loss=0.09_F.pt'))
        model.V.load_state_dict(torch.load('/work/jprieto/data/trachoma/train/classification/Analysis_Set_202208/stacks_v9_efficientnet_v2s_dot_recall_w8/epoch=26-val_loss=0.09_V.pt'))
        model.P.load_state_dict(torch.load('/work/jprieto/data/trachoma/train/classification/Analysis_Set_202208/stacks_v9_efficientnet_v2s_dot_recall_w8/epoch=26-val_loss=0.09_P.pt'))
        print("FINISH LOADING!")


    elif args.nn == "efficientnet_v2s_stacks_sigdot":
        model = EfficientnetV2sStacksSigDot(args, out_features=unique_classes.shape[0], class_weights=unique_class_weights, model_patches=model_patches)
    elif args.nn == "mobilenet_v2_stacks":
        model = MobileNetV2Stacks(args, out_features=unique_classes.shape[0], class_weights=unique_class_weights, model_patches=model_patches)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=30, verbose=True, mode="min")

    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)    

    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
        devices=torch.cuda.device_count(), 
        accelerator="gpu", 
        strategy=DDPStrategy(find_unused_parameters=False),
        log_every_n_steps=args.log_every_n_steps
    )
    trainer.fit(model, datamodule=ttdata, ckpt_path=args.model)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='TT classification Training')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--model', help='Model path to continue training', type=str, default=None)
    parser.add_argument('--model_patches', help='Trianed Patches model', type=str, default=None)
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)    
    parser.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=50)    
    parser.add_argument('--out', help='Output', type=str, default="./")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=64)
    parser.add_argument('--nn', help='Type of neural network', type=str, default="efficientnet_v2s_stacks")    
    parser.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    parser.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="classification_efficientnet_v2s")


    args = parser.parse_args()

    main(args)
