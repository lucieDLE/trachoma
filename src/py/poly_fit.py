import itk
import numpy as np
import argparse
import matplotlib.pyplot as plt
from utils import *


def main(args):
	img = itk.imread(args.img)
	label = itk.imread(args.label)

	img_np = itk.GetArrayViewFromImage(img)
	label_np = itk.GetArrayViewFromImage(label)

	y, x = np.where(label_np == args.label_num)

	z = np.polyfit(x, y, 3)
	poly = np.poly1d(z)

	neigborhood = int(args.size/2)

	min_x = np.min(x) + neigborhood
	max_x = np.max(x) - neigborhood
	max_y = np.max(y) - neigborhood

	dx = (max_x - min_x)/args.num_samples

	out_stack = []

	x_values = []
	y_values = []
	for i in range(args.num_samples):
		x = min_x + i*dx
		y = poly(x)

		x_values.append(x)
		y_values.append(y)

		start_x = min(max(int(x) - neigborhood, 0), max_x  - neigborhood)
		start_y = min(max(int(y) - neigborhood, 0), max_y  - neigborhood)

		end_x = start_x + args.size
		end_y = start_y + args.size

		out_stack.append(img_np[start_y:end_y,start_x:end_x,:])

	out_stack = np.array(out_stack)
	out_img = GetImage(out_stack)

	# itk.imwrite(out_img, args.out)

	plt.imshow(img_np)
	plt.imshow(label_np, interpolation='nearest', alpha=0.4)

	plt.plot(x_values, y_values)
	plt.plot(x, y)
	plt.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Predict an input with a trained neural network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--img', type=str, help='Input rgb image', required=True)
	parser.add_argument('--label', type=str, help='Input label image', required=True)
	parser.add_argument('--label_num', type=int, help='Label number for polyfit', default=3)
	parser.add_argument('--num_samples', type=int, help='Output number of samples', default=64)
	parser.add_argument('--size', type=int, help='Output size', default=256)
	parser.add_argument('--out', type=str, help='Output image stack', default="out.nrrd")

	args = parser.parse_args()
	main(args)