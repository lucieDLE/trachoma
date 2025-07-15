import math
import numpy as np 
from torchvision.ops import boxes as box_ops
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import pandas as pd 
from torchvision import models
from torchvision import transforms
import torchmetrics
from utils import FocalLoss
import monai
from evaluation import *
from visualization import select_eyelid_seg, filter_indices_on_segmentation_mask
from monai.transforms import (
    ToTensord
)
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.ops import nms
import lightning.pytorch as pl
from sklearn.metrics import f1_score, balanced_accuracy_score
class TTUSeg(nn.Module):
    def __init__(self, unet):
        super(TTUSeg, self).__init__()
        self.unet = unet

    def forward(self, x):
        x = torch.unsqueeze(x, 0)
        x = torch.permute(x, (0, 3, 1, 2))
        x = self.unet(x)
        x = torch.permute(x, (0, 2, 3, 1))
        x = torch.argmax(x, dim=-1, keepdim=True)
        x = x.type(torch.uint8)
        x = torch.squeeze(x, 0)
        return x

class TTUSegTorch(nn.Module):
    def __init__(self, unet):
        super(TTUSegTorch, self).__init__()
        self.unet = unet

    def forward(self, x):        
        x = self.unet(x)        
        x = torch.argmax(x, dim=1, keepdim=True)
        x = x.type(torch.uint8)
        return x

class TTUNet(pl.LightningModule):
    def __init__(self, out_channels=4, **kwargs):
        super(TTUNet, self).__init__()        
        
        self.save_hyperparameters()

        if hasattr(self.hparams, "ce_weight"):
            self.loss = monai.losses.DiceCELoss(include_background=False, to_onehot_y=True, softmax=True, ce_weight=torch.tensor(self.hparams.ce_weight), lambda_dice=1.0, lambda_ce=1.0)
        else:
            self.loss = monai.losses.DiceLoss(include_background=False, softmax=True, to_onehot_y=True)
        

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.out_channels)

        self.model = monai.networks.nets.UNet(spatial_dims=2, in_channels=3, out_channels=self.hparams.out_channels, channels=(16, 32, 64, 128, 256, 512, 1024), strides=(2, 2, 2, 2, 2, 2), num_res_units=4)

        self.metric = DiceMetric(include_background=True, reduction="mean")   

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def forward(self, x):
        
        x = self.model(x)
        x = torch.argmax(x, dim=1, keepdim=True)
        return x

    def training_step(self, train_batch, batch_idx):
        
        x = train_batch["img"]
        y = train_batch["seg"]
        
        y = y.to(torch.int64)
        x = self.model(x)

        loss = self.loss(x, y)
        
        batch_size = x.shape[0]
        self.log('train_loss', loss, batch_size=batch_size)        

        # x = torch.argmax(x, dim=1, keepdim=True)
        # self.accuracy(x.reshape(-1, 1), y.reshape(-1, 1))
        # self.log("train_acc", self.accuracy, batch_size=batch_size)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch["img"]
        y = val_batch["seg"]
        
        y = y.to(torch.int64)
        x = self.model(x)
        
        loss = self.loss(x, y)
        
        batch_size = x.shape[0]
        self.log('val_loss', loss, batch_size=batch_size)
        
        # x = torch.argmax(x, dim=1, keepdim=True)
        # self.accuracy(x.reshape(-1, 1), y.reshape(-1, 1))
        # self.log("val_acc", self.accuracy, batch_size=batch_size)

    def predict_step(self, images):
        return  self(images)

class TTRCNN(pl.LightningModule):
    def __init__(self, device='cuda', **kwargs):
        super(TTRCNN, self).__init__()        
        
        self.save_hyperparameters()
        self.model = models.detection.maskrcnn_resnet50_fpn(weights=models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
                                                            rpn_fg_iou_thresh=0.5,
                                                            rpn_nms_thr = 0.6,
                                                            detections_per_img = 4,
                                                            box_detections_per_img = 4,
                                                            rpn_post_nms_top_n_train=50,
                                                            rpn_post_nms_top_n_test=50,
                                                            )

        self.num_classes = self.hparams.out_features
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, self.num_classes)

        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, self.num_classes)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
        return optimizer

    def compute_bb_mask(self, seg, pad=0.5):
      shape = seg.shape[1:]
      bbx = []
      masks = []
      for i in range(1,4): ## remove the background 
        
        ij = torch.argwhere(seg.squeeze() == i)
        mask = torch.zeros_like(seg)
        mask[ seg == i] = 1

        if len(ij) > 0:
            bb = torch.tensor([0, 0, 0, 0])# xmin, ymin, xmax, ymax

            bb[0] = torch.clip(torch.min(ij[:,1]) - 5, 0, shape[1])
            bb[1] = torch.clip(torch.min(ij[:,0]) - 5, 0, shape[0])

            bb[2] = torch.clip(torch.max(ij[:,1]) + 5, 0, shape[1])
            bb[3] = torch.clip(torch.max(ij[:,0]) + 5, 0, shape[0])
        else:
            print("problem")
            bb = torch.tensor([0, 0, shape[1], shape[0]])# xmin, ymin, xmax, ymax
        
        bbx.append(bb.unsqueeze(0))
        masks.append(mask)
      return torch.cat(bbx), torch.cat(masks)

    def forward(self, data, mode='train'):
        images = data['img'].to(self.device)
        if mode == 'train':

            segs = data['seg'].to(self.device)
            self.model.train()
            targets = []
            for seg in segs:
                d = {}
                box, masks = self.compute_bb_mask(seg, pad=0.01)
                d['boxes'] = box.to(self.device)
                d['labels'] = torch.tensor([1,2,3]).to(self.device)
                d['masks'] = masks.to(self.device)
                targets.append(d)

            losses = self.model(images, targets)
            return losses

        if mode == 'val': # get the boxes and losses
            self.model.train()
            with torch.no_grad():
                segs = data['seg'].to(self.device)
                targets = []
                for seg in segs:
                    d = {}
                    box, masks = self.compute_bb_mask(seg, pad=0.01)
                    d['boxes'] = box.to(self.device)
                    d['labels'] = torch.tensor([0,1,2,3]).to(self.device)
                    d['masks'] = masks.to(self.device)
                    targets.append(d)

                losses = self.model(images, targets)
                
                self.model.eval()
                preds = self.model(images)
                self.model.train()
                return [losses, preds]

        elif mode == 'test': # prediction
            self.model.eval()
            output = self.model(images)

            return output

    def training_step(self, train_batch, batch_idx):
        
        x = train_batch["img"]
        y = train_batch["seg"]
        
        loss_dict = self(train_batch)
        for w, n in zip(self.hparams.loss_weights, loss_dict.keys()):
            loss = loss_dict[n]
            total_loss += w*loss
            self.log(f'train/{n}', loss, sync_dist=True)
        self.log('train_loss', total_loss,sync_dist=True,batch_size=self.hparams.batch_size)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch["img"]
        y = val_batch["seg"]
        
        loss_dict, preds = self(val_batch, mode='val')
        total_loss = 0
        for w, n in zip(self.hparams.loss_weights, loss_dict.keys()):
            loss = loss_dict[n]
            total_loss += w*loss
            self.log(f'val/{n}', loss, sync_dist=True)
  
        self.log('val_loss', total_loss,sync_dist=True,batch_size=self.hparams.batch_size)

    def predict_step(self, images):
        test_batch = {'img': images}

        outputs = self(test_batch, mode='test')

        seg_stack = []
        for out in outputs:
            masks = out['masks'].cpu().detach()
            seg = self.compute_segmentation(masks, out['labels'])
            seg_stack.append(seg.unsqueeze(0))
        return torch.cat(seg_stack)

    def compute_segmentation(self, masks, labels,thr=0.3):
        ## need a smoothing steps I think, very harsh lines
        labels = labels.cpu().detach().numpy()
        seg_mask = torch.zeros_like(masks[0]) 
        for i in range(masks.shape[0]):
            seg_mask[ masks[i]> thr ] = labels[i]
        return seg_mask
            

class RandomRotate(nn.Module):
    def __init__(self, degrees, keys, interpolation, prob=0.5):        
        self.degrees = degrees
        self.keys = keys
        self.interpolation = interpolation
        self.prob = prob

    def __call__(self, X):

        if self.prob > torch.rand(1):

            angle = torch.rand(1)*self.degrees
            
            for k, interp in zip(self.keys, self.interpolation):
                X[k] = transforms.functional.rotate(X[k], angle=angle.item(), interpolation=interp)

            return X

        return X

class RandomFlip(nn.Module):
    def __init__(self, keys, prob=0.5):
        self.keys = keys   
        self.prob = prob

    def __call__(self, X):

        if self.prob > torch.rand(1):
            for k in self.keys:
                X[k] = transforms.functional.hflip(X[k])
            return X

        return X

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=1280, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, x):
        return self.model(x)

class MobileYOLO(pl.LightningModule):
    def __init__(self, **kwargs):
        super(MobileYOLO, self).__init__()        
        
        self.save_hyperparameters()

        self.model = nn.Sequential(
            models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features,            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            ProjectionHead(input_dim=1280, hidden_dim=1280, output_dim=4)#h,w,i,j
            )

        self.resize_bb = transforms.Resize(self.hparams.size_bb)

        self.train_transform = transforms.Compose(
            [
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5)
            ]
        )

        self.spatial_train_transform = transforms.Compose(
            [
                RandomRotate(degrees=90, keys=["img", "seg"], interpolation=[transforms.functional.InterpolationMode.NEAREST, transforms.functional.InterpolationMode.NEAREST], prob=0.5), 
                RandomFlip(keys=["img", "seg"], prob=0.5)
            ]
        )

        self.loss_fn = torch.nn.SmoothL1Loss()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def compute_bb(self, seg, label=3, pad=0):
    
        shape = seg.shape[1:]
        
        ij = torch.argwhere(seg.squeeze() == label)

        bb = torch.tensor([0, 0, 0, 0])# xmin, ymin, xmax, ymax

        if len(ij) > 0:

            bb[0] = torch.clip(torch.min(ij[:,1]) - shape[1]*pad, 0, shape[1])
            bb[1] = torch.clip(torch.min(ij[:,0]) - shape[0]*pad, 0, shape[0])
            bb[2] = torch.clip(torch.max(ij[:,1]) + shape[1]*pad, 0, shape[1])
            bb[3] = torch.clip(torch.max(ij[:,0]) + shape[0]*pad, 0, shape[0])
        
        return bb
    
    def forward(self, X):        
        return self.model(self.resize_bb(X).detach().clone())

    def training_step(self, train_batch, batch_idx):

        train_batch = self.spatial_train_transform(train_batch)
        
        Y_bb = torch.stack([self.compute_bb(seg, pad=self.hparams.pad) for seg in train_batch["seg"]]).to(self.device)
        
        x_bb = self(self.train_transform(train_batch["img"]))

        loss = self.loss_fn(x_bb, Y_bb.to(torch.float32))
                
        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):

        Y_bb = torch.stack([self.compute_bb(seg, pad=self.hparams.pad) for seg in val_batch["seg"]]).to(self.device)

        x_bb = self(val_batch["img"])

        loss = self.loss_fn(x_bb, Y_bb.to(torch.float32))
                
        self.log('val_loss', loss, sync_dist=True)
        

    def test_step(self, test_batch, batch_idx):
        Y_bb = torch.stack([self.compute_bb(seg, pad=self.hparams.pad) for seg in test_batch["seg"]]).to(self.device)

        x_bb = self(test_batch["img"])

        loss = self.loss_fn(x_bb, Y_bb.to(torch.float32))
                
        self.log('test_loss', loss, sync_dist=True)



# Define custom RoIHeads with class weights. Need forward pass to access needed data.
# Re-apply functions to get the labels. Needed because shapes are differents -> class logits has a 
# shape matching the number of regions proposed (n=100) and not the number of boxes passed in the batch 
class CustomRoIHeads(RoIHeads):
    def __init__(self, *args, hparams=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.hparams = hparams
        self.class_weights = torch.tensor(self.hparams.class_weights, dtype=torch.float32, device='cuda')
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.1)

    def forward(self, features, proposals, image_shapes, targets=None):
        result = []

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result= []
        losses = {}
        if self.training:
            _, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            labels = torch.cat(labels, dim=0)

            loss_classifier = self.ce_loss(class_logits, labels)

            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append({"boxes": boxes[i], "labels": labels[i], "scores": scores[i],})

        return result, losses

class FasterTTRCNN(pl.LightningModule):
    def __init__(self, device='cuda', **kwargs):
        super(FasterTTRCNN, self).__init__()
        
        self.save_hyperparameters()


        self.model = models.detection.fasterrcnn_resnet50_fpn_v2(weights=models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                                                             min_size=512,
                                                             max_size=1024,
                                                             rpn_fg_iou_thresh=0.4,
                                                             rpn_bg_iou_thresh = 0.1,
                                                             rpn_nms_thr = 0.2,
                                                             rpn_post_nms_top_n_train=200,
                                                             rpn_post_nms_top_n_test=100,
                                                             box_detections_per_img=25,
                                                             )


        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.out_features) #conflict

        self.model.roi_heads = CustomRoIHeads(self.model.roi_heads.box_roi_pool,
                                              self.model.roi_heads.box_head,
                                              models.detection.faster_rcnn.FastRCNNPredictor(in_features, self.hparams.out_features),
                                              fg_iou_thresh=0.6,
                                              bg_iou_thresh=0.4,
                                              batch_size_per_image=512,
                                              bbox_reg_weights=None,
                                              positive_fraction=0.15,
                                              score_thresh=0.05,
                                              nms_thresh=0.5,
                                              detections_per_img=25,
                                              hparams = self.hparams,
                                              )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        return optimizer

    def forward(self, images, targets=None, mode='test'):
        images = images.to(self.device)
        if mode == 'train':
            self.model.train()
            losses = self.model(images, targets)
            return losses

        elif mode == 'val': # get the boxes and losses
            with torch.no_grad():
                self.model.train()
                losses = self.model(images, targets)
                
                self.model.eval()
                preds = self.model(images)
                self.model.train()
                return [losses, preds]
        else:
            self.model.eval()
            output = self.model(images)

            return output

    def training_step(self, train_batch, batch_idx):
        
        loss_dict = self(train_batch[0], train_batch[1], mode='train')
        total_loss = 0
        for w, n in zip(self.hparams.loss_weights, loss_dict.keys()):
            loss = loss_dict[n]
            total_loss += w*loss
            self.log(f'train/{n}', loss, sync_dist=True)
        self.log('train_loss', total_loss,sync_dist=True,batch_size=self.hparams.batch_size)
                
        return total_loss

    def validation_step(self, val_batch, batch_idx):
                
        loss_dict, preds = self(val_batch[0], val_batch[1], mode='val')
        total_loss = 0
        for w, n in zip(self.hparams.loss_weights, loss_dict.keys()):
            loss = loss_dict[n]
            total_loss += w*loss
            self.log(f'val/{n}', loss, sync_dist=True)
        self.log('val_loss', total_loss,sync_dist=True,batch_size=self.hparams.batch_size)

        f1_macro, f1_per_class, balanced_acc = self.evaluate_accuracy(val_batch[0], val_batch[1], preds)

        # Log with PyTorch Lightning + Neptune
        self.log("val_acc/f1_macro", f1_macro)
        self.log("val_acc/balanced_acc", balanced_acc)
        
        for i, score in enumerate(f1_per_class):
            self.log(f"val_acc/f1_class_{i}", score)

    def predict_step(self, images):

        return self(images, mode='test')

    def evaluate_accuracy(self, imgs, targets, preds):
        gt, pred = [], []

        for p, t in zip(preds, targets):

            gt_indices = nms(t['boxes'], torch.ones_like(t['boxes'][:,0]), iou_threshold=1.0) ## iou as args
            t['boxes'] = t['boxes'][gt_indices].cpu().detach()
            t['labels'] = t['labels'][gt_indices].cpu().detach()  
            t['mask'] = t['mask'].cpu().detach()

            gt.append(gt_eye_outcome(t['labels']))

            ### -- preds -- ###
            eyelid_seg = select_eyelid_seg(t['mask'])
            p = filter_indices_on_segmentation_mask(eyelid_seg, p, overlap_threshold=0.5)

            preds = {}
            for k in p.keys():
                preds[k] = p[k].cpu().detach()

            if len(preds['scores']) >= 1:
                preds = process_predictions(preds)
            # thr = p['scores'].mean() - 2*p['scores'].std()
            # keep = p['scores'] > thr
            # p = filter_targets_indices(p, keep)

            pred.append(eye_level_outcome(preds, imgs.shape[2:]))

        gt = torch.tensor(gt)
        pred = torch.tensor(pred)

        df_eval = pd.DataFrame(data={'gt':gt, 'pred':pred})
        df_sel = df_eval.loc[ df_eval['gt'] != -1]
        df_sel = df_sel.loc[ df_sel['pred'] != -1]

        df_sel = df_sel[['pred', 'gt']] -1

        # Compute metrics
        f1_macro = f1_score(df_sel['gt'], df_sel['pred'], average='macro')
        f1_per_class = f1_score(df_sel['gt'], df_sel['pred'], average=None)
        balanced_acc = balanced_accuracy_score(df_sel['gt'], df_sel['pred'])

        return f1_macro, f1_per_class, balanced_acc

class TTRPN(nn.Module):
    def __init__(self, faster):
        super(TTRPN, self).__init__()
        self.faster = faster

    def forward(self, images, targets=None):

        original_image_sizes: list[tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.faster.transform(images, targets)

        features = self.faster.backbone(images.tensors)
        proposals, proposal_loss = self.faster.rpn(images, features, targets)
        return proposals

class TTRoidHead(nn.Module):
    def __init__(self, faster):
        super(TTRoidHead, self).__init__()
        self.faster = faster
        self.images_shapes = None
        self.og_sizes = None

    def forward(self, images, proposals, targets=None):

        original_image_sizes: list[tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.faster.transform(images, targets)
        self.images_shapes = images.image_sizes
        self.og_sizes = original_image_sizes

        features = self.faster.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        roi_out = self.faster.roi_heads.box_roi_pool(features, proposals, images.image_sizes)
        box_features = self.faster.roi_heads.box_head(roi_out)
        class_logits, box_regression = self.faster.roi_heads.box_predictor(box_features)
        # boxes, scores, labels = self.faster.roi_heads.postprocess_detections(class_logits, box_regression, proposals, images.image_sizes)

        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.faster.roi_heads.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, images.image_sizes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            scores, labels = torch.max(scores, dim=1)
            batch_indices = torch.arange(boxes.size(0), device=boxes.device)
            
            boxes = boxes[batch_indices, labels]
            
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)


        detections = []
        num_images = len(all_labels)
        for i in range(num_images):
            detections.append(
                {
                    "boxes": all_boxes[i],
                    "labels": all_labels[i],
                    "scores": all_scores[i],
                }
            )
        
        return detections[0]['boxes'], detections[0]['labels'], detections[0]['scores']
    
# errors when exporting from onnx to tflite at inference
class TTFullModel(nn.Module):
    def __init__(self, faster):
        super(TTFullModel, self).__init__()
        self.faster = faster
        self.images_shapes = None
        self.og_sizes = None

    def forward(self, images, targets=None):

        original_image_sizes: list[tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.faster.transform(images, targets)
        self.images_shapes = images.image_sizes
        self.og_sizes = original_image_sizes

        features = self.faster.backbone(images.tensors)
        proposals, proposal_loss = self.faster.rpn(images, features, targets)

        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        roi_out = self.faster.roi_heads.box_roi_pool(features, proposals, images.image_sizes)
        box_features = self.faster.roi_heads.box_head(roi_out)
        class_logits, box_regression = self.faster.roi_heads.box_predictor(box_features)

        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.faster.roi_heads.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, images.image_sizes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            scores, labels = torch.max(scores, dim=1)
            batch_indices = torch.arange(boxes.size(0), device=boxes.device)
            
            boxes = boxes[batch_indices, labels]
            
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)


        detections = []
        num_images = len(all_labels)
        for i in range(num_images):
            detections.append(
                {
                    "boxes": all_boxes[i],
                    "labels": all_labels[i],
                    "scores": all_scores[i],
                }
            )
        
        return detections[0]['boxes'], detections[0]['labels'], detections[0]['scores']