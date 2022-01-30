import numpy as np
import itk

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