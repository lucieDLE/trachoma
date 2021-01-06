import numpy as np
import tensorflow as tf
import argparse
import os
import itk
import sys
from utils import *

def main(args):
	print("Tensorflow version:", tf.__version__)

	saved_model_path = args.model
	out_name = args.out

	model = tf.keras.models.load_model(saved_model_path, custom_objects={'tf': tf})
	model.summary()

	img = itk.imread(args.img)
	img_np = itk.GetArrayViewFromImage(img)

	prediction = model.predict(img_np.reshape((1,) + img_np.shape))

	out_img = GetImage(prediction[0])

	itk.imwrite(out_img, args.out)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Predict a label map from a natural image of an eye', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	input_group = parser.add_argument_group('Input parameters')
	input_group.add_argument('--img', type=str, help='Input image for prediction')

	model_group = parser.add_argument_group('Model group')
	model_group.add_argument('--model', help='Directory of trained model in saved model format')

	out_group = parser.add_argument_group('Output parameters')
	out_group.add_argument('--out', type=str, help='Output image name', default="out.nrrd")

	args = parser.parse_args()

	main(args)