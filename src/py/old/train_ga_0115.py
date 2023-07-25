import os
import time
from collections import Counter
import re
import math
from math import ceil
from datetime import datetime
import argparse

import pandas as pd
import numpy as np
import itk
import nrrd

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Sampler
from torchvision import transforms as T
import torchvision.models as models
#from torchvision.transforms import ToTensor
from torchvision.transforms import Pad
from torchvision.transforms import Resize, ToPILImage

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data.distributed

from torch.distributions.utils import probs_to_logits
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed import ReduceOp
from apex.parallel import DistributedDataParallel as DDP
from apex import amp

from monai.data import ITKReader, PILReader
from monai.transforms import (
    ToTensor, LoadImage, Lambda, AddChannel, RepeatChannel, ScaleIntensityRange, RandSpatialCrop,
    Resized, Compose, BorderPad, NormalizeIntensity
)
from monai.config import print_config


dataset_dir = "/mnt/raid/C1_ML_Analysis/"
output_mount = "/mnt/famli_netapp_shared/C1_ML_Analysis/train_out/"


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
       
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class TransformFrames:
    def __init__(self, training=True):
        if training:
            self.transform = Compose([
    #        Lambda(func=random_choice), 
            AddChannel(),
            BorderPad(spatial_border=[-1, 32, 32]),
            RandSpatialCrop(roi_size=[-1, 256, 256], random_size=False),
            #Flip(spatial_axis=3),
            #ScaleIntensityRange(0.0, 255.0, 0, 1.0),
            #RepeatChannel(repeats=3),
            ToTensor(),
            #Lambda(lambda x: torch.transpose(x, 0, 1)),
            #NormalizeIntensity(subtrahend=np.array([0.485, 0.456, 0.406]), divisor=np.array([0.229, 0.224, 0.225])),
            #ToTensor(),
            #normalize 
            ])
        else:
            self.transform = Compose([
        #    Lambda(func=random_choice), 
            AddChannel(),
            #BorderPad(spatial_border=[-1, 24, 24]),
            #RandSpatialCrop(roi_size=[-1, 256, 256], random_size=False),
            #Flip(spatial_axis=3),
            #ScaleIntensityRange(0.0, 255.0, 0, 1.0),
            #RepeatChannel(repeats=3),
            ToTensor(),
            #Lambda(lambda x: torch.transpose(x, 0, 1)),
            #NormalizeIntensity(subtrahend=np.array([0.485, 0.456, 0.406]), divisor=np.array([0.229, 0.224, 0.225])),
            #ToTensor(),
            #normalize 
            ])

    def __call__(self, x):
        x = self.transform(x)
        x = torch.transpose(x, 0, 1)
        return x

class ITKImageDataset(Dataset):
    def __init__(self, df, mount_point,
                transform=None, num_sample_frames=None, device='cpu'):
        self.df = df
        self.mount = mount_point
        self.targets = self.df.iloc[:, 1].to_numpy()
        self.transform = transform
        self.num_sample_frames = num_sample_frames
        self.device = device

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = self.df.iloc[idx, 0]
        ga = self.df.iloc[idx, 1]
        try:
            img, header = nrrd.read(os.path.join(self.mount, img_path), index_order='C')
            img = torch.tensor(img, dtype=torch.float, device=self.device)
            assert(len(img.shape) == 3)
            assert(img.shape[1] == 256)
            assert(img.shape[2] == 256)
        except:
            print("Error reading cine: " + img_path)
            img = torch.zeros(200, 256, 256)
        if self.num_sample_frames:
            idx = torch.randint(img.size(0), (self.num_sample_frames,))
            img = img[idx]
        if self.transform:
            img = self.transform(img)
        # normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # im_list = [normalize(m) for m in torch.unbind(img, dim=0)]
        # img = torch.stack(im_list, dim=0)
        return img, np.array([ga])

class ITKImageDatasetByID(Dataset):
    def __init__(self, df, ga_col, mount_point,
                transform=None, cache=False, device='cpu'):
        self.df = df
        self.mount = mount_point
        self.ga_col = ga_col
        self.transform = transform
        ga_map = self.df[["study_id_uuid", "ga_boe"]].drop_duplicates().reset_index(drop=True)
        self.study_ids = ga_map.study_id_uuid.values
        self.targets = ga_map.ga_boe.to_numpy()
        self.cache = cache
        self.data_map = dict()
        self.device = device

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        relevant_rows = self.df.loc[self.df.study_id_uuid == study_id,:]
        #shuffle rows
        relevant_rows = relevant_rows.sample(frac=1).reset_index(drop=True)
        ga = relevant_rows.loc[0, self.ga_col]
        seq_im_array = []
        for i, row in relevant_rows.iterrows():
            img_path = row['uuid_path']
            try:
                if self.cache and (img_path in self.data_map):
                    img = self.data_map[img_path]
                else:
                    img, header = nrrd.read(os.path.join(self.mount, img_path), index_order='C')
                    if self.cache:
                        self.data_map[img_path] = img
                img = torch.tensor(img, dtype=torch.float, device=self.device)
                assert(len(img.shape) == 3)
                assert(img.shape[1] == 256)
                assert(img.shape[2] == 256)
            except:
                print("Error reading cine: " + img_path)
                img = torch.zeros(1, 256, 256)
            seq_im_array.append(img)
        im_array = torch.cat(seq_im_array, dim=0)
        if self.transform:
            im_array = self.transform(im_array)
        # normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # im_list = [normalize(m) for m in torch.unbind(im_array, dim=0)]
        # img = torch.stack(im_list, dim=0)
        img = im_array
        return img, np.array([ga])

class DistributedWeightedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, replacement=True, shuffle=True):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement
        self.shuffle = shuffle

    def calculate_weights(self, targets):
        ga_week_counter = Counter()
        targets = targets.numpy()
        for ga in targets:
            ga_week_counter[ga // 7] += 1

        weights = np.ones(len(targets))
        for i in range(len(weights)):
            ga = targets[i]
            weights[i] = 1 / ga_week_counter[ga // 7]

        samples_weight = torch.tensor(weights)
        return samples_weight

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # get targets (you can alternatively pass them in __init__, if this op is expensive)
        targets = self.dataset.targets
        # select only the wanted targets for this subsample
        targets = torch.tensor(targets)[indices]
        assert len(targets) == self.num_samples
        # randomly sample this subset, producing balanced classes
        weights = self.calculate_weights(targets)
        subsample_balanced_indicies = torch.multinomial(weights, self.num_samples, self.replacement)
        # now map these target indicies back to the original dataset index...
        dataset_indices = torch.tensor(indices)[subsample_balanced_indicies]

        return iter(dataset_indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class SelfAttention(nn.Module):
    def __init__(self, in_units, out_units):
        super(SelfAttention, self).__init__()
        self.W1 = nn.Linear(in_units, out_units)
        self.V = nn.Linear(out_units, 1)

    def forward(self, query, values):        

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(nn.Tanh()(self.W1(query)))

        # min_score = tf.reduce_min(tf.math.top_k(tf.reshape(score, [-1, tf.shape(score)[1]]), k=self.k, sorted=False, name=None)[0], axis=1, keepdims=True)
        # min_score = tf.reshape(min_score, [-1, 1, 1])
        # score_mask = tf.greater_equal(score, min_score)
        # score_mask = tf.cast(score_mask, tf.float32)
        # attention_weights = tf.multiply(tf.exp(score), score_mask) / tf.reduce_sum(tf.multiply(tf.exp(score), score_mask), axis=1, keepdims=True)

        # attention_weights shape == (batch_size, max_length, 1)
        score = nn.Sigmoid()(score)
        sum_score = torch.sum(score, 1, keepdim=True)
        attention_weights = score / sum_score

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, attention_weights

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module
 
    def forward(self, input_seq):
        assert len(input_seq.size()) > 2
 
        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps

        size = input_seq.size()

        batch_size = size[0]
        time_steps = size[1]

        size_reshape = [batch_size*time_steps] + list(size[2:])
        reshaped_input = input_seq.contiguous().view(size_reshape)
 
        output = self.module(reshaped_input)
        
        output_size = output.size()
        output_size = [batch_size, time_steps] + list(output_size[1:])
        output = output.contiguous().view(output_size)

        return output


class GA_Net(nn.Module):
    def __init__(self):
        super(GA_Net, self).__init__()

        cnn = models.efficientnet_b0(pretrained=True)
        cnn.classifier = Identity()

        self.TimeDistributed = TimeDistributed(cnn)
        self.WV = nn.Linear(1280, 128)
        self.Attention = SelfAttention(1280, 64)
        self.Prediction = nn.Linear(128, 1)
 
    def forward(self, x):

        x = self.TimeDistributed(x)

        x_v = self.WV(x)

        x_a, w_a = self.Attention(x, x_v)

        x = self.Prediction(x_a)

        return x, w_a

class GA_Net_features(nn.Module):
    def __init__(self, cnn_pretrained):
        super(GA_Net_features, self).__init__()

        cnn = cnn_pretrained

        self.TimeDistributed = TimeDistributed(cnn)

    def forward(self, x):
        
        x = self.TimeDistributed(x)

        return x
       
class GA_Net_attn_output(nn.Module):
    def __init__(self):
        super(GA_Net_attn_output, self).__init__()

        self.WV = nn.Linear(1280, 128)
        self.Attention = SelfAttention(1280, 64)
        self.Prediction = nn.Linear(128, 1)
        
 
    def forward(self, x):

        x_v = self.WV(x)
        x_a, w_a = self.Attention(x, x_v)

        x = self.Prediction(x_a)

        return x, w_a


def train(gpu, args):
    ############################################################
    rank = args.nr * args.gpus + gpu                              
    dist.init_process_group(                                   
        backend='nccl',                                         
        init_method='env://',                                   
        world_size=args.world_size,                              
        rank=rank                                               
    )
    torch.manual_seed(0)                                                          
    ############################################################
    ####### Model #############
    model = GA_Net()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    ###############################################################
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu])
    ###############################################################
    batch_size = 12
    # define loss function (criterion) and optimizer
    loss_fn = nn.L1Loss().cuda(gpu)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # Data loading code
    train_df = pd.read_csv(os.path.join(dataset_dir, 'CSV_files', 'uuid_train_new_256.csv'))
    print(np.unique(train_df.tag, return_counts = True))
    train_df = train_df.loc[:,["uuid_path", "ga_boe"]].reset_index(drop=True)
    training_data = ITKImageDataset(train_df, mount_point=dataset_dir,
                                    transform=TransformFrames(training=True), num_sample_frames = 50, device='cpu')
    train_sampler = DistributedWeightedSampler(training_data, num_replicas=args.world_size, 
                                        rank=rank, replacement=True, shuffle=True)
    train_dataloader = DataLoader(
        training_data, batch_size=batch_size, sampler=train_sampler, num_workers=6, pin_memory=True)
    val_df = pd.read_csv(os.path.join(dataset_dir, 'CSV_files', 'uuid_valid_new_256.csv'))
    val_df = val_df.loc[:,["study_id_uuid","uuid_path", "ga_boe"]].reset_index(drop=True)
    val_data = ITKImageDatasetByID(val_df, mount_point=dataset_dir,
                                ga_col="ga_boe",
                                transform=TransformFrames(training=False),
                                cache=False, device='cpu')
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_data, num_replicas=args.world_size, rank=rank)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=1,
                                              sampler=val_sampler, num_workers=6, pin_memory=True,
                                              persistent_workers=True)
    num_epochs = 200
    early_stop = EarlyStopping(patience=10, verbose=True, 
                                path=os.path.join(output_mount, 'model/model_ga_0115.pt'))
    # Keep track of losses
    f_train_epoch_loss_history = open(os.path.join(output_mount,"log/train_model_ga_0115" + ".txt"),"w", buffering=1)
    f_validation_epoch_history = open(os.path.join(output_mount,"log/valid_model_ga_0115" + ".txt"),"w", buffering=1)

    n_batch = 5000
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float)[None, None, :, None, None].cuda()
    sd = torch.as_tensor([0.229, 0.224, 0.225], dtype=torch.float)[None, None, :, None, None].cuda()
    l = 1.0
    beta = 3.0

    def entropy(weights):
        min_real = torch.finfo(torch.float32).min
        logits = probs_to_logits(weights)
        logits = torch.clamp(logits, min=min_real)
        p_log_p = logits * weights
        return -p_log_p.sum(-1)
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        num_batches = 0
        while num_batches != n_batch:
            for batch, (X, y) in enumerate(train_dataloader):
                num_batches += 1
                batch_size = X.size(0)

                X = X.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                X.div_(255)
                X = X.repeat_interleave(3,dim=2)
                X.sub_(mean).div_(sd)
                x, w_a = model(X)
                L_e = torch.mean(torch.distributions.Categorical(w_a.squeeze(2)).entropy())
                loss = loss_fn(x, y) + (l * L_e)
                # loss = loss_fn(x, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if L_e > beta:
                    l = 1.01 * l
                else:
                    l = 0.99 * l
                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{len(training_data):>5d}]")
                if num_batches == n_batch:
                    break
        train_loss = running_loss / num_batches
        print(f"average epoch loss: {train_loss:>7f}  [{epoch:>5d}/{num_epochs:>5d}]")
        train_loss = torch.tensor([train_loss]).cuda()
        dist.all_reduce(train_loss, op=ReduceOp.SUM)
        train_loss = train_loss.cpu().item() / args.world_size
        if rank == 0:
            f_train_epoch_loss_history.write("Epoch: " + str(epoch) + '\n')
            f_train_epoch_loss_history.write("Num batches: " + str(num_batches) + '\n')
            f_train_epoch_loss_history.write(str(train_loss) + '\n')
        #################################
        dist.barrier()
        if rank == 0:
            f_train_epoch_loss_history.write("Training Loop Time (minutes): " + str(((time.time() - epoch_start)/60.0)) +'\n')
        #################################
        model.eval()
        model_features = GA_Net_features(model.module.TimeDistributed.module)
        model_attn_output = GA_Net_attn_output()
        model_attn_output.WV = model.module.WV
        model_attn_output.Attention = model.module.Attention
        model_attn_output.Prediction = model.module.Prediction
        model_features.eval()
        model_attn_output.eval()
        model_features.cuda()
        model_attn_output.cuda()
        with torch.no_grad():
            running_loss = 0.0
            for batch, (X, y) in enumerate(val_dataloader):
                y = y.cuda()
                batch_size = X.size(0)
                features_list = []
                for x_chunk in torch.split(X, 500, dim=1):
                    if torch.cuda.is_available():
                        x_chunk = x_chunk.cuda(non_blocking=True)
                    x_chunk.div_(255)
                    x_chunk = x_chunk.repeat_interleave(3,dim=2)
                    x_chunk.sub_(mean).div_(sd)
                    features_chunk = model_features(x_chunk)
                    features_list.append(features_chunk)
                features = torch.cat(features_list, dim=1)
                out, w_a = model_attn_output(features)
                loss = loss_fn(out, y)
                running_loss += loss.item()

        val_loss = torch.tensor([running_loss / len(val_sampler)]).cuda()
        dist.all_reduce(val_loss, op=ReduceOp.SUM)
        val_loss = val_loss.cpu().item() / args.world_size
        if rank == 0:
            f_validation_epoch_history.write("epoch " + str(epoch) + '\n')
            f_validation_epoch_history.write(str(val_loss) + '\n')
            early_stop(val_loss, model.module)
            f_train_epoch_loss_history.write("Time (minutes): " + str(((time.time() - epoch_start)/60.0)) +'\n')
            if early_stop.early_stop:
                early_stop_indicator = torch.tensor([1.0]).cuda()
            else:
                early_stop_indicator = torch.tensor([0.0]).cuda()
        else:
            early_stop_indicator = torch.tensor([0.0]).cuda()
        dist.all_reduce(early_stop_indicator, op=ReduceOp.SUM)
        if early_stop_indicator.cpu() == torch.tensor([1.0]):
            print("Early stopping")            
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    #########################################################
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'         
    mp.spawn(train, nprocs=args.gpus, args=(args,))
    #########################################################

if __name__ == '__main__':
    main()