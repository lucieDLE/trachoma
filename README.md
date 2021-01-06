# trachoma
Trachoma prediction from natural images.

## Prediction of label map

Input an rgb image of dimension 256x256. The original input image should be resampled to this size.


```
python3 src/py/predict.py --img input_256.jpg --model ./useg_3labels_256/ --out input_labelmap_256.nrrd
```

## Fit of polynomial to output label

Once the prediction is done, we proceed to generate a stack of images by following the best polynomial fit to the segmented region. 
In this case, our region of interest is given by label number 3. 
The output from the previous step must be resampled to a higher resolution using nearest neighbor interpolation.


```
python3 src/py/poly_fit.py --img input_2048.jpg --size 512 --label input_labelmap_2048.nrrd --out input_stacked.nrrd
```

![Poly fit](./doc/poly_fit.png)
