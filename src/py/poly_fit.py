import SimpleITK as sitk
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def main(args):
	img = sitk.ReadImage(args.img)
	seg = sitk.ReadImage(args.seg)
	img_np = sitk.GetArrayFromImage(img)
	seg_np = sitk.GetArrayFromImage(seg)

	out_stack = poly_fit(img_np, seg_np, args.label_num, args.size, args.num_samples)
	
	out_img = sitk.GetImageFromArray(out_stack)
	sitk.WriteImage(out_img, args.out)

	if args.view:

		plt.imshow(img_np)

		cmap = ListedColormap(["black", "tab:red", "tab:green", "tab:blue"])
		plt.imshow(seg_np, cmap=cmap, interpolation='nearest', alpha=0.4)

		plt.plot(x_values, y_values, linewidth=2, color="red")
		plt.plot(x, y)
		plt.show()

def poly_fit(img_np, seg_np, label_num=3, size=256, num_samples=64):	
	
	y, x = np.where(seg_np == label_num)	

	z = np.polyfit(x, y, 3)
	poly = np.poly1d(z)

	neigborhood = int(size/2)

	min_x = np.min(x) + neigborhood
	max_x = np.max(x) - neigborhood
	max_y = np.max(y) - neigborhood

	dx = (max_x - min_x)/num_samples

	out_stack = []

	x_values = []
	y_values = []
	for i in range(num_samples):
		x = min_x + i*dx
		y = poly(x)

		x_values.append(x)
		y_values.append(y)

		start_x = min(max(int(x) - neigborhood, 0), max_x  - neigborhood)
		if start_x < 0:
			start_x = 0
		start_y = min(max(int(y) - neigborhood, 0), max_y  - neigborhood)
		if(start_y < 0):
			start_y = 0

		end_x = start_x + size
		end_y = start_y + size

		crop_np = img_np[start_y:end_y,start_x:end_x,:]

		out_stack.append(crop_np)

	out_stack = np.array(out_stack)

	return out_stack


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Predict an input with a trained neural network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--img', type=str, help='Input rgb image', required=True)
	parser.add_argument('--seg', type=str, help='Input label image', required=True)
	parser.add_argument('--label_num', type=int, help='Label number for polyfit', default=3)
	parser.add_argument('--num_samples', type=int, help='Output number of samples', default=64)
	parser.add_argument('--size', type=int, help='Output size', default=256)
	parser.add_argument('--out', type=str, help='Output image stack', default="out.nrrd")
	parser.add_argument('--view', type=bool, help='View the fit', default=False)

	args = parser.parse_args()
	main(args)