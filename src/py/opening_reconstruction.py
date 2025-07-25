import pandas 
import os 
import numpy 
import matplotlib.pyplot as plt
import argparse
import SimpleITK as sitk
import numpy as np


def main(args):

  ## do stuff

  seg_paths = os.listdir(args.indir)

  for seg_path in seg_paths:
    sitk_seg = sitk.ReadImage(os.path.join(args.indir,seg_path))
    opened_seg = sitk.OpeningByReconstruction(sitk_seg, [20, 20, 20])

    writer = sitk.ImageFileWriter()
    writer.SetFileName(os.path.join(args.outdir, seg_path))
    writer.UseCompressionOn()
    writer.Execute(opened_seg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, help='Input directory', required=True)
    parser.add_argument('--outdir', type=str, help='Output directory', required=True)

    args = parser.parse_args()

    main(args)