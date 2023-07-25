import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch

from nets.classification import EfficientnetV2s, MobileNetV2
from loaders.tt_dataset import TTDataModule, TrainTransforms, EvalTransforms

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

from pytorch_lightning.callbacks import QuantizationAwareTraining
from torch.utils.mobile_optimizer import optimize_for_mobile

from sklearn.utils import class_weight

def replace_last(str, old, new):
    if old not in str:
        return str
    idx = str.rfind(old)
    return str[:idx] + new + str[idx+len(old):]

def main(args):

    # train_fn = os.path.join(args.mount_point, 'Analysis_Set_20220422', 'trachoma_bsl_mtss_besrat_field_patches_train_20220422_fold4_train.csv')
    # valid_fn = os.path.join(args.mount_point, 'Analysis_Set_20220422', 'trachoma_bsl_mtss_besrat_field_patches_train_20220422_fold4_test.csv')
    # test_fn = os.path.join(args.mount_point, 'Analysis_Set_20220422', 'trachoma_bsl_mtss_besrat_field_patches_test_20220422.csv')

    # train_fn = os.path.join(args.mount_point, 'Analysis_Set_20220422', 'trachoma_bsl_mtss_besrat_field_patches_train_20220422_clean.csv')
    # valid_fn = os.path.join(args.mount_point, 'Analysis_Set_20220422', 'trachoma_bsl_mtss_besrat_field_patches_train_20220422_clean_eval.csv')
    # test_fn = os.path.join(args.mount_point, 'Analysis_Set_20220422', 'trachoma_bsl_mtss_besrat_field_patches_test_20220422.csv')

    train_fn = os.path.join(args.mount_point, 'Analysis_Set_202208', 'trachoma_bsl_mtss_besrat_field_patches_train_202208_train.csv')
    valid_fn = os.path.join(args.mount_point, 'Analysis_Set_202208', 'trachoma_bsl_mtss_besrat_field_patches_train_202208_eval.csv')
    test_fn = os.path.join(args.mount_point, 'Analysis_Set_202208', 'trachoma_bsl_mtss_besrat_field_patches_test_202208.csv')

    df_train = pd.read_csv(train_fn)
    df_train.drop(df_train[df_train['patch_class'].isin(['Probable Epilation', 'Probable TT', 'Unknown'])].index, inplace = True)
    
    df_train["patch_class"] = df_train["patch_class"].replace({'Healthy': 0, 'Epilation': 1, 'TT': 2})
    unique_classes = np.sort(np.unique(df_train["patch_class"]))
    # print(df_train["patch_class"])
    # print(unique_classes)
    # unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train["patch_class"]))
    unique_class_weights = np.array(class_weight.compute_class_weight(class_weight={0: 1, 1: 4, 2: 8}, classes=unique_classes, y=df_train["patch_class"]))
    

    class_replace = {}
    for cn, cl in enumerate(unique_classes):
        class_replace[cl] = cn
    print(unique_classes, unique_class_weights, class_replace)

    df_train["patch_class"] = df_train["patch_class"].replace(class_replace).astype(int)

    df_val = pd.read_csv(valid_fn)    
    df_val.drop(df_val[df_val['patch_class'].isin(['Probable Epilation', 'Probable TT', 'Unknown'])].index, inplace = True)
    df_val["patch_class"] = df_val["patch_class"].replace({'Healthy': 0, 'Epilation': 1, 'TT': 2})
    # df_val["patch_class"] = df_val["patch_class"].replace(class_replace).astype(int)
    
    df_test = pd.read_csv(test_fn)
    df_test.drop(df_test[df_test['patch_class'].isin(['Probable Epilation', 'Probable TT', 'Unknown'])].index, inplace = True)
    df_test["patch_class"] = df_test["patch_class"].replace({'Healthy': 0, 'Epilation': 1, 'TT': 2})
    # df_test["patch_class"] = df_test["patch_class"].replace(class_replace)

    
    ttdata = TTDataModule(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column='image', class_column="patch_class")


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )

    if args.nn == "efficientnet_v2s":
        model = EfficientnetV2s(args, out_features=unique_classes.shape[0], class_weights=unique_class_weights)
    elif args.nn == "mobilenet_v2":
        model = MobileNetV2(args, out_features=unique_classes.shape[0], class_weights=unique_class_weights)

    # model.model.load_state_dict(torch.load("/work/jprieto/data/trachoma/train/train_patch_mobilnet_v2_torch/Analysis_Set_202208/patches_qat/epoch=76-val_loss=0.11.pt"))
    
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=30, verbose=True, mode="min")

    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)    


    modules_to_fuse = []

    for name, module in model.named_modules():

        try:
            if (isinstance(module, torch.nn.Conv2d) 
                and isinstance(model.get_submodule(replace_last(name, '.0', '.1')), torch.nn.BatchNorm2d) 
                and  isinstance(model.get_submodule(replace_last(name, '.0', '.2')), torch.nn.ReLU)):
                modules_to_fuse.append((name, replace_last(name, '.0', '.1'), replace_last(name, '.0', '.2')))
        except:
            continue

    #CALLBACK IS BULLSHIT FOR NOW -> QuantizationAwareTraining(qconfig='qnnpack', observer_type="histogram", modules_to_fuse=modules_to_fuse)
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

    trainer.test(datamodule=ttdata)

    # tsmodel = model.to_torchscript()

    # tsmodel_optimized = optimize_for_mobile(tsmodel)
    # tsmodel_optimized._save_for_lite_interpreter(os.path.join(args.out, "features.ptl"))


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='TT classification Training')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--model', help='Model path to continue training', type=str, default=None)
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)    
    parser.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=50)    
    parser.add_argument('--out', help='Output', type=str, default="./")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=256)
    parser.add_argument('--nn', help='Type of neural network', type=str, default="efficientnet_v2s")    
    parser.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    parser.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="classification_efficientnet_v2s")


    args = parser.parse_args()

    main(args)
