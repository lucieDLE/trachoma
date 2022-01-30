import SimpleITK as sitk
import argparse
import numpy as np


def main(args):
	img = sitk.ReadImage(args.img)

	writer = sitk.ImageFileWriter()
	writer.SetFileName(args.out)
	writer.UseCompressionOn()
	writer.Execute(img)
	


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--img', type=str, help='Input image', required=True)
	parser.add_argument('--out', type=str, help='Output image', required=True)

	args = parser.parse_args()

	main(args)