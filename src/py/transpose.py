import SimpleITK as sitk
import argparse
import numpy as np


def main(args):
	img = sitk.ReadImage(args.img)

	permute = sitk.PermuteAxesImageFilter()
	permute.SetOrder(args.axes)
	img_out = permute.Execute(img)

	sitk.WriteImage(img_out, args.out)	


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--img', type=str, help='Input image', required=True)
	parser.add_argument('--axes', type=int, nargs="+", help='Axes order', required=True)
	parser.add_argument('--out', type=str, help='Output image', required=True)

	args = parser.parse_args()

	main(args)