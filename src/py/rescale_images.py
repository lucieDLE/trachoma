#!/usr/bin/env python

"""
Script to resample an input image or all images in a given directory
to a target size.

Inputs:
    image_path | input_dir : Either a single image or or directory of images
    output_dir : Output directory destination
    target_size: Target size (one number: image will be resampled to a square)
    resample_method : 0 - nearest neighborhood (default), 1 - linear
Output:
    Resampled images with a suffix of <image_name>_ <target_size>.<original image suffix>
    These will be stored in output_dir

Prereqs: needs SimpleITK

Author: Hina Shah
"""

from argparse import ArgumentParser
from pathlib import Path, PurePosixPath
import shutil
import SimpleITK as sitk
from numpy import squeeze

# -------------------------------------------------------------------
def rescale_image(image_path, out_image_dir, target_size = 256, resample_method=0):

    ext = image_path.suffix
    if ext.lower() in ['.dcm', '.dicom']:
        raise FileError('DICOM Extensions are not supported yet')

    im = sitk.ReadImage(str(image_path))
    dimension = im.GetDimension()

    if ext.lower() == '.nrrd' and dimension == 3:
        imspacing = im.GetSpacing()
        imorigin = im.GetOrigin()
        imdirection = im.GetDirection()

        # Need to do the following to squeeze the extra dimension - mostly because nrrd's are supposed to be 3d images by default
        nparray = sitk.GetArrayFromImage(im)
        nparray = squeeze(nparray)
        im = sitk.GetImageFromArray(nparray)
        im.SetSpacing(imspacing[0:2])
        im.SetOrigin(imorigin[0:2])
        im.SetDirection(imdirection[0:2] + imdirection[3:5])

    new_size = [target_size]*im.GetDimension()
    reference_image = sitk.Image(new_size, im.GetPixelIDValue())
    reference_image.SetOrigin(im.GetOrigin())
    reference_image.SetDirection(im.GetDirection())
    reference_image.SetSpacing([sz*spc/nsz for nsz,sz,spc in zip(new_size, im.GetSize(), im.GetSpacing())])

    resample_method = 0 if resample_method not in [0,1] else resample_method
    sitk_resample_method = sitk.sitkLinear if resample_method == 1 else sitk.sitkNearestNeighbor

    # Resample
    im_resampled = sitk.Resample(im, reference_image, sitk.Transform(), sitk_resample_method)

    # Write
    out_image_name = image_path.name.replace( ext, '_' + str(target_size) + ext )
    sitk.WriteImage(im_resampled, str(out_image_dir/out_image_name))

# -------------------------------------------------------------------
def main(args):

    # Create image list
    image_path_list = []
    if args.image:
        image_path_list = [ Path(args.image)]
    else:
        # i.e. input directory was given.
        if not args.image_suffix:
            print("Don't know which image type to look for. Supply a valid suffix")
            return
        suffix =  '.' + args.image_suffix if args.image_suffix[0] != '.' else args.image_suffix
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            print('Input directory {} is not a directory'.format(input_dir))
        image_path_list = list( input_dir.glob('*' + suffix)  )

    print('Found {} images'.format(len(image_path_list)))
    if len(image_path_list) == 0:
        return

    # Create output directory if it does not exist
    output_dir_path = Path(args.out_dir)
    if not output_dir_path.is_dir():
        output_dir_path.mkdir(parents=True)

    # Call resampling
    for image_path in image_path_list:
        try:
            rescale_image(image_path, output_dir_path, args.target_size, args.resample_method)
        except Exception as e:
            print('ERROR processing path: {} \n\n {}'.format(image_path, e))

    print('----- DONE ------')

# -------------------------------------------------------------------
if __name__=="__main__":
    parser = ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Path to the image to be resampled')
    input_group.add_argument('--input_dir', type=str, help='Directory with all the images to be resampled')
    parser.add_argument('--image_suffix', type=str, default='.jpg', help='Suffix of the images to look for in the directory. This is required if input_dir is given. jpg is default')
    parser.add_argument('--out_dir', type=str, help='Output directory path', required=True)
    parser.add_argument('--target_size', type=int, default=256, help='Target size of the square resampled image')
    parser.add_argument('--resample_method', type=int, default=0, help='Resampling method to be used: \n 0 - nearest neighborhood (default) \n 1 - linear')

    args = parser.parse_args()

    main(args)