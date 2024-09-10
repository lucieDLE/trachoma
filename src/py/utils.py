import numpy as np
import torch 
import torch.nn as nn

def GetImage(img_np, ctype = 'float'):
	img_np_shape = np.shape(img_np)
	ComponentType = itk.ctype(ctype)

	Dimension = img_np.ndim - 1
	PixelDimension = img_np.shape[-1]
	print("Dimension:", Dimension, "PixelDimension:", PixelDimension)

	if Dimension == 1:
		OutputImageType = itk.VectorImage[ComponentType, 2]
	else:
		OutputImageType = itk.VectorImage[ComponentType, Dimension]
	
	out_img = OutputImageType.New()
	out_img.SetNumberOfComponentsPerPixel(PixelDimension)

	size = itk.Size[OutputImageType.GetImageDimension()]()
	size.Fill(1)
	
	prediction_shape = list(img_np.shape[0:-1])
	prediction_shape.reverse()

	if Dimension == 1:
		size[1] = prediction_shape[0]
	else:
		for i, s in enumerate(prediction_shape):
			size[i] = s

	index = itk.Index[OutputImageType.GetImageDimension()]()
	index.Fill(0)

	RegionType = itk.ImageRegion[OutputImageType.GetImageDimension()]
	region = RegionType()
	region.SetIndex(index)
	region.SetSize(size)

	out_img.SetRegions(region)
	out_img.Allocate()

	out_img_np = itk.GetArrayViewFromImage(out_img)
	out_img_np.setfield(img_np.reshape(out_img_np.shape), out_img_np.dtype)

	return out_img

import pdb
def mixup_img_seg(x, seg, y, alpha=0.5):
	batch_size = x.shape[0]

	yhot = nn.functional.one_hot(y, num_classes=2)
	
	lam = np.random.beta(alpha, alpha, (batch_size,)).astype(dtype=np.float32)
	lam = torch.from_numpy(lam).to(x.device)

	index = torch.randperm(batch_size)
	lam = lam.view(batch_size, *[1]*x[0].dim())

	x_perm = torch.stack([x[idx,...] for idx in index])
	seg_perm = torch.stack([seg[idx,...] for idx in index])
	
	mixed_x = lam * x + (1 - lam) * x_perm
	mixed_seg = lam * seg + (1 - lam) * seg_perm

	lam = lam.view(batch_size, 1)
	mixed_y = lam * yhot + (1 - lam) * yhot[index, ...]


	return  {"img":mixed_x, "seg":mixed_seg, "class":mixed_y}


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', weights =None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weights = weights

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none', weight=self.weights)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss