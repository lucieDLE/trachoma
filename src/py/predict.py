import numpy as np
import tensorflow as tf
import argparse
import importlib
import os
from datetime import datetime
import json
import glob
import itk
import sys
import pandas
import csv

def image_read(filename, args):
  print("Reading:", filename)
  if(args.image_dimension == 1):
    if(args.pixel_dimension != -1):
      ImageType = itk.Image[itk.Vector[itk.F, args.pixel_dimension], 2]
    else:
      ImageType = itk.VectorImage[itk.F, 2]
  else:
    if(args.pixel_dimension == 1):
      ImageType = itk.Image[itk.F, args.image_dimension]
    elif(args.pixel_dimension != -1):
      ImageType = itk.Image[itk.Vector[itk.F, args.pixel_dimension], args.image_dimension]
    else:
      ImageType = itk.VectorImage[itk.F, args.image_dimension]
    
  img_read = itk.ImageFileReader[ImageType].New(FileName=filename)
  img_read.Update()
  img = img_read.GetOutput()
  
  img_np = itk.GetArrayViewFromImage(img).astype(float)

  if(args.image_dimension == 1):
    img_np = img_np.reshape([s for s in img_np.shape if s != 1])

  print(img_np.shape, img.GetImageDimension())
  if args.flip_x:
    print("Flip x:")
    if img.GetImageDimension() == 3:
      img_np = np.flip(img_np, axis=2)
    else:
      img_np = np.flip(img_np, axis=1)

  if args.flip_y:
    print("Flip y")
    if img.GetImageDimension() == 3:
      img_np = np.flip(img_np, axis=1)
    else:
      img_np = np.flip(img_np, axis=0)

  # if args.flip_z:
  #   print("Flip z")
  #   if img.GetImageDimension() == 3:
  #     img_np = np.flip(img_np, axis=0)

  # Put the shape of the image in the json object if it does not exists. This is done for global information
  tf_img_shape = list(img_np.shape)
  if(tf_img_shape[0] == 1 and img.GetImageDimension() > 2):
    # If the first component is 1 we remove it. It means that is a 2D image but was saved as 3D
    tf_img_shape = tf_img_shape[1:]

  # This is the number of channels, if the number of components is 1, it is not included in the image shape
  # If it has more than one component, it is included in the shape, that's why we have to add the 1
  if(img.GetNumberOfComponentsPerPixel() == 1):
    tf_img_shape = tf_img_shape + [1]

  if args.shuffle:
    np.random.shuffle(img_np)
  
  if not args.batch_prediction:
    tf_img_shape = [1] + tf_img_shape

  return img, np.reshape(img_np, tf_img_shape)


def image_save(img_obj, prediction, img):

  Dimension = prediction.ndim - 1
  PixelDimension = prediction.shape[-1]
  print("Dimension:", Dimension, "PixelDimension:", PixelDimension)
  
  if(PixelDimension < 7):
    if(PixelDimension >= 3 and os.path.splitext(img_obj["out"])[1] not in ['.jpg', '.png']):
      ComponentType = itk.ctype('float')
      PixelType = itk.Vector[ComponentType, PixelDimension]
    elif(PixelDimension == 3):
      PixelType = itk.RGBPixel.UC
      prediction = np.absolute(prediction)
      prediction = np.around(prediction).astype(np.uint16)
    else:
      PixelType = itk.ctype('float')

    OutputImageType = itk.Image[PixelType, Dimension]
    out_img = OutputImageType.New()

  else:

    ComponentType = itk.ctype('float')
    if Dimension == 1 or Dimension == 0:
      OutputImageType = itk.VectorImage[ComponentType, 2]
    else:
      OutputImageType = itk.VectorImage[ComponentType, Dimension]

    out_img = OutputImageType.New()
    out_img.SetNumberOfComponentsPerPixel(PixelDimension)
  
  size = itk.Size[OutputImageType.GetImageDimension()]()
  size.Fill(1)

  print("Prediction shape:", prediction.shape)
  prediction_shape = list(prediction.shape[0:-1])
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
  
  # out_img.SetRegions(img.GetLargestPossibleRegion())
  out_img.SetRegions(region)
  if(Dimension == img.GetImageDimension()):
    out_img.SetDirection(img.GetDirection())
    out_img.SetOrigin(img.GetOrigin())
    out_img.SetSpacing(img.GetSpacing())
  out_img.Allocate()

  out_img_np = itk.GetArrayViewFromImage(out_img)
  out_img_np.setfield(np.reshape(prediction, out_img_np.shape), out_img_np.dtype)

  print("Writing:", img_obj["out"])
  writer = itk.ImageFileWriter.New(FileName=img_obj["out"], Input=out_img)
  writer.UseCompressionOn()
  writer.Update()

def main(args):
  saved_model_path = args.model
  out_name = args.out
  out_ext = args.out_ext
  prediction_type = args.prediction_type

  data_description = None
  class_obj = None
  data_description_filename = os.path.join(saved_model_path, "data_description.json")
  if os.path.exists(data_description_filename):
    with open(data_description_filename, "r") as f:
      data_description = json.load(f)
      if "enumerate" in data_description and prediction_type is None or prediction_type == "class":
        prediction_type = "class"
        class_obj = {}
        enumerate_obj = data_description[data_description["enumerate"]]["class"]["enumerate"]
        print(enumerate_obj)
        for key in enumerate_obj:
          class_obj[enumerate_obj[key]] = key 
        print(class_obj)
      if "prediction_type" in data_description:
        prediction_type = data_description["prediction_type"]


  filenames_df = None
  if(args.img):
    filenames_arr = []    
    print('image_name', args.img)
    fobj = {}
    fobj["img"] = args.img
    if(prediction_type == "img" or prediction_type == "seg"):
      fobj["out"] = out_name
      if args.ow or not os.path.exists(fobj["out"]):
        filenames_arr.append(fobj)
    else:
      filenames_arr.append(fobj)

    filenames_df = pandas.DataFrame(filenames_arr)

  else:

    filenames_arr = []
    image_filenames = []
    replace_dir_name = ''
    if(args.dir):
      replace_dir_name = args.dir
      normpath = os.path.normpath("/".join([args.dir, '**', '*']))
      for img in glob.iglob(normpath, recursive=True):
        if os.path.isfile(img) and True in [ext in img for ext in [".nrrd", ".nii", ".nii.gz", ".mhd", ".dcm", ".DCM", ".jpg", ".png"]]:
          if(args.use_realpath):
            image_filenames.append(os.path.realpath(img))
          else:
            image_filenames.append(img)

      for img in image_filenames:
        fobj = {}
        fobj["img"] = img
        if(prediction_type == "img" or prediction_type == "seg"):
          image_dir_filename = img.replace(replace_dir_name, '')
          if(out_ext):
            image_dir_filename = os.path.splitext(image_dir_filename)[0] +  out_ext

          if(args.out_basename):
            image_dir_filename = os.path.basename(image_dir_filename)
            
          fobj["out"] = os.path.normpath("/".join([out_name, image_dir_filename]))

          if not os.path.exists(os.path.dirname(fobj["out"])):
            os.makedirs(os.path.dirname(fobj["out"]))

        if args.ow or ("out" in fobj and not os.path.exists(fobj["out"])) or prediction_type == "class" or prediction_type == "scalar":
          filenames_arr.append(fobj)

      filenames_df = pandas.DataFrame(filenames_arr)

    elif(args.csv):
      replace_dir_name = args.csv_root_path
      with open(args.csv) as csvfile:
        df = pandas.read_csv(csvfile)
        df.rename({args.csv_column: "img"}, inplace=True)

        img_out = []

        if(prediction_type == "img" or prediction_type == "seg"):
          for index, row in df.iterrows():
            img = row["img"]
          
            image_dir_filename = img.replace(replace_dir_name, '')
            if(out_ext):
              image_dir_filename = os.path.splitext(image_dir_filename)[0] +  out_ext

            if(args.out_basename):
              image_dir_filename = os.path.basename(image_dir_filename)
              
            out_fn = os.path.normpath("/".join([out_name, image_dir_filename]))

            if not os.path.exists(os.path.dirname(out_fn)):
              os.makedirs(os.path.dirname(out_fn))

            if args.ow or not os.path.exists(out_fn):
              img_out.append(out_fn)
            else:
              img_out.append(None)
        
        if len(img_out) > 0:
          df["out"] = img_out
          df = df[df["out"].notnull()]

        filenames_df = df

  with tf.device('/device:GPU:' + str(args.gpu)):
    model = tf.keras.models.load_model(saved_model_path, custom_objects={'tf': tf})
    model.summary()

    if(args.pre_process):
    	pre_process = eval(args.pre_process)

    prediction_arr = []
    for i, img_obj in filenames_df.iterrows():
      try:
        img, img_np = image_read(img_obj["img"], args)
        
        if(args.pre_process):
        	print("pre_process")
        	img_np = pre_process(img_np)
        prediction = model.predict(img_np)

        if(args.batch_prediction):
          prediction = prediction
        elif prediction_type == "array":
          prediction = [np.ravel(p).tolist() for p in prediction]
        else:
          prediction = prediction[0]

        if(prediction_type == "img" or prediction_type == "seg"):
          image_save(img_obj, prediction, img)
        elif(prediction_type == "class"):
          if class_obj is not None:
            argmax = np.argmax(prediction)
            prediction_arr.append(class_obj[argmax])
            print("prediction", prediction, "class", class_obj[argmax])
          else:
            if len(prediction) == 1:
              prediction = int(np.round(prediction[0]))
            prediction_arr.append(prediction)
            print("prediction", prediction, img_obj["class"])
        elif(prediction_type == "scalar"):
          prediction_arr.append(prediction.tolist()[0])
          print("prediction", prediction)
        elif(prediction_type == "array"):
          prediction_arr.append(prediction)
      except Exception as e:
        prediction_arr.append(None)
        print(e, file=sys.stderr)

    if prediction_type == "class" or prediction_type == "scalar" or prediction_type == "array":
      filenames_df["prediction"] = prediction_arr

      print("Writing:", args.out)
      filenames_df.to_csv(args.out, index=False)
    else:
      out_csv = args.out + ".csv"
      print("Writing:", out_csv)
      filenames_df = filenames_df.rename({"out": "prediction"})
      filenames_df.to_csv(out_csv, index=False)

if __name__ == "__main__":

  print("Tensorflow version:", tf.__version__)

  parser = argparse.ArgumentParser(description='Predict an input with a trained neural network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  input_group = parser.add_argument_group('Input parameters')

  in_group = input_group.add_mutually_exclusive_group(required=True)
  in_group.add_argument('--img', type=str, help='Input image for prediction')
  in_group.add_argument('--dir', type=str, help='Directory with images for prediction')
  in_group.add_argument('--csv', type=str, help='CSV file with images. ')
  input_group.add_argument('--use_realpath', type=int, default=0, help='Use the real_path, i.e., eliminate any symbolink link before processing')
  input_group.add_argument('--csv_column', type=str, default='img', help='CSV column name (Only used if flag csv is used)')
  input_group.add_argument('--csv_root_path', type=str, default='', help='Replaces a root path directory to empty, this is use to recreate a directory structure in the output directory, otherwise, the output name will be the name in the csv (only if csv flag is used)')

  image_group = parser.add_argument_group('Image parameters')
  image_group.add_argument('--image_dimension', type=int, help='Image dimension', default=2)
  image_group.add_argument('--pixel_dimension', type=int, help='Pixel dimension. It will try to guess it by default.', default=-1)
  image_group.add_argument('--batch_prediction', type=bool, help='The image is 3D/sequence but the network is for 2D images. i.e., it will use the 3rd dimension as a batch', default=False)
  image_group.add_argument('--shuffle', type=bool, help='Shuffle image in the first dimension', default=False)
  image_group.add_argument('--flip_x', type=bool, help='Flip image x axis', default=False)
  image_group.add_argument('--flip_y', type=bool, help='Flip image y axis', default=False)

  model_group = parser.add_argument_group('Model group')
  model_group.add_argument('--model', help='Directory of saved model format')
  model_group.add_argument('--pre_process', help='Use a keras pre_processing step before doing the prediction, ex: tf.keras.applications.vgg19.preprocess_input', default=None)
  model_group.add_argument('--prediction_type', help='Type of prediction. img, seg, class, scalar', default="img")
  model_group.add_argument('--gpu', help='GPU index', type=int, default=0)
  model_group.add_argument('--predict_cross', help='Predict all folds from a cross validation training. Use --csv flag with name of csv without "_fold0_test.csv", the --model directory is the same one you used on "--save_model" when exporting the trained models. --out is an output directory in this case', type=int, default=0)

  out_group = parser.add_argument_group('Output parameters')
  out_group.add_argument('--out', type=str, help='Output image, csv, or directory. If --dir flag is used the output image name will be the <Directory set in out flag>/<image filename in directory dir>', default="out")
  out_group.add_argument('--out_ext', type=str, help='Output extension for images', default='.nrrd')
  out_group.add_argument('--out_basename', type=bool, default=False, help='Keeps only the filename for the output, i.e, does not create a directory structure for the output image filename')
  out_group.add_argument('--ow', type=int, help='Overwrite outputs', default=1)

  args = parser.parse_args()

  if args.predict_cross and args.csv:
    csv_in = args.csv

    csv_folds = os.path.splitext(csv_in)[0] + '*fold*_test.csv'
    out_csv = args.out
    model_dir = args.model
    
    for csv_fn in glob.iglob(csv_folds):
      args.csv = csv_fn
      out_name = os.path.basename(csv_fn).replace("_test.csv", "_prediction.csv")
      args.out = os.path.join(os.path.splitext(out_csv)[0], out_name)
      model_name = os.path.basename(csv_fn).replace("_test.csv", "_train")
      args.model = os.path.join(model_dir, model_name)
      print("Predict cross validation fold.")
      main(args)

    print("Saving full df.")
    full_df = None
    for csv_fn in glob.iglob(csv_folds):
      in_name = os.path.join(os.path.splitext(out_csv)[0], os.path.basename(csv_fn).replace("_test.csv", "_prediction.csv"))
      print(in_name)
      if full_df is None:
        full_df = pandas.read_csv(in_name)
      else:
        full_df = full_df.append(pandas.read_csv(in_name), ignore_index = True)
    out_name_full_df = os.path.splitext(csv_in)[0] + "_prediction_all.csv"
    print("Writing:", out_name_full_df)
    full_df.to_csv(out_name_full_df, index=False)
      

  else:
    main(args)