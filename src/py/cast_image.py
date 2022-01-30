import SimpleITK as sitk
import argparse
import numpy as np


def main(args):
	img = sitk.ReadImage(args.img)
	img_np = sitk.GetArrayFromImage(img)

	if args.mul:
		img_np = img_np.astype(float)*args.mul
	
	img_np = img_np.astype(args.type)

	out_img = sitk.GetImageFromArray(img_np)

	writer = sitk.ImageFileWriter()
	writer.SetFileName(args.out)
	writer.UseCompressionOn()
	writer.Execute(out_img)
	


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--img', type=str, help='Input image', required=True)
	parser.add_argument('--mul', type=float, help='Multiply', default=None)
	parser.add_argument('--out', type=str, help='Output image', default="out.nrrd")
	parser.add_argument('--type', type=str, help='Output type', default="ubyte")

	args = parser.parse_args()

	main(args)