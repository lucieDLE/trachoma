from pytorch_lightning.callbacks import Callback
import torchvision
import torch

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class SegImageLogger(Callback):
    def __init__(self, num_images=12, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            
            img1 = batch["img"]
            img2 = batch["seg"]
            
            max_num_image = min(img1.shape[0], self.num_images)
            grid_img1 = torchvision.utils.make_grid(img1)
            trainer.logger.experiment.add_image('img', grid_img1[0:max_num_image], pl_module.global_step)
            
            grid_img2 = torchvision.utils.make_grid(img2/torch.max(img2))
            trainer.logger.experiment.add_image('seg', grid_img2[0:max_num_image], pl_module.global_step) 

            with torch.no_grad():
                x_hat = pl_module(img1)
                grid_img3 = torchvision.utils.make_grid(x_hat/torch.max(x_hat))
                trainer.logger.experiment.add_image('x_hat', grid_img3[0:max_num_image], pl_module.global_step)

class SegYOLOImageLogger(Callback):
    def __init__(self, num_images=2, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:

            with torch.no_grad():
                x_bb, X_patches = pl_module(batch)

                X_patches = X_patches[0:self.num_images].cpu()

                for i in range(self.num_images):
                    x_p = X_patches[i]

                    grid_p = torchvision.utils.make_grid(x_p, nrow=pl_module.hparams.num_patches)
                    trainer.logger.experiment.add_image('grid_p{i}'.format(i=i), grid_p, pl_module.global_step)


class StackImageLogger(Callback):
    def __init__(self, num_images=2, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:

            with torch.no_grad():
                x_bb, X_patches = pl_module(batch)

                X_patches = X_patches[0:self.num_images].cpu()

                for i in range(self.num_images):
                    x_p = X_patches[i]

                    grid_p = torchvision.utils.make_grid(x_p, nrow=pl_module.hparams.num_patches)
                    trainer.logger.experiment.add_image('grid_p{i}'.format(i=i), grid_p, pl_module.global_step)