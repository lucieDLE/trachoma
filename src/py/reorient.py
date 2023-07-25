import SimpleITK as sitk
import argparse
import numpy as np
from PIL import Image, ExifTags

def reorient(img):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        
        exif = img.getexif()

        if exif[orientation] == 3:
            return img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            return img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            return img.rotate(90, expand=True)

        return img
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass

def main(args):
    img = Image.open(args.img)
    img = reorient(img)
    img.save(args.out)
    img.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, help='Input image', required=True)
    parser.add_argument('--out', type=str, help='Output image', required=True)

    args = parser.parse_args()

    main(args)