import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import SequenceEnqueuer
from tensorflow.keras.utils import OrderedEnqueuer
import keras
import keras.backend as K

import pickle  
import SimpleITK as sitk

from scipy import ndimage

import pickle

from skimage.transform import resize

from tqdm import tqdm
import os 

class Features(tf.keras.layers.Layer):
    def __init__(self):
        super(Features, self).__init__()

        self.efficient = tf.keras.applications.EfficientNetV2S(include_top=False, weights='imagenet', input_tensor=tf.keras.Input(shape=[448, 448, 3]), pooling=None)
        
        self.center_crop = tf.keras.layers.CenterCrop(448, 448)        
        self.conv = layers.Conv2D(512, (2, 2), strides=(2, 2))
        self.avg = layers.GlobalAveragePooling2D()        

    def compute_output_shape(self, input_shape):
        return (None, 512)


    def call(self, x, training=True):
        
        x = self.center_crop(x)        
        x = self.efficient(x)
        x = self.conv(x)
        x = self.avg(x)

        return x

class TTModelPatch(tf.keras.Model):
    def __init__(self):
        super(TTModelPatch, self).__init__()

        self.features = Features()
        self.P = layers.Dense(3, activation='softmax', name='predictions')
        
    def call(self, x):

        x_f = self.features(x)
        x = self.P(x_f)

        return x


class Attention(tf.keras.layers.Layer):
    def __init__(self, units, w_units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(w_units)

    def call(self, query, values):        

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(query)))
        
        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, score

class TTModel(tf.keras.Model):
    def __init__(self, features = None):
        super(TTModel, self).__init__()

        self.features = Features()        

        self.TD = layers.TimeDistributed(self.features)
        self.R = layers.Reshape((-1, 512))

        self.V = layers.Dense(256)
        self.A = Attention(128, 1)        
        self.P = layers.Dense(2, activation='softmax', name='predictions')
        
    def call(self, x):

        x = self.TD(x)
        x = self.R(x)

        x_v = self.V(x)
        x_a, x_s = self.A(x, x_v)
        
        x = self.P(x_a)
        x_v_p = self.P(x_v)

        # return x, x_a, x_s, x_v, x_v_p
        return x

class TTModelPred(tf.keras.Model):
    def __init__(self, features = None):
        super(TTModelPred, self).__init__()

        self.features = Features()
        
        self.R = layers.Reshape((-1, 512))

        self.V = layers.Dense(256)        
        self.P = layers.Dense(2, activation='softmax', name='predictions')
        
    def call(self, x):

        x = self.features(x)
        x = self.R(x)

        x = self.V(x)
        x = self.P(x)

        return x


class DatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
            
        try:
            row = self.df.loc[idx]
            # img = os.path.join("/work/jprieto/data/remote/EGower/hinashah/Analyses_Set_20220321_Images_stacks/", row["image"].replace(".jpg", ".nrrd"))
            img = os.path.join("hinashah/hinashah_applist/jimma_tis_may_2022_discordant_NOTT_stacks/", row["image"].replace(".jpg", ".nrrd"))
            sev = row["tt sev"]

            img_np = sitk.GetArrayFromImage(sitk.ReadImage(img))

            t, xs, ys, _ = img_np.shape
            xo = (xs - 448)//2
            yo = (ys - 448)//2
            img_np = img_np[:,xo:xo + 448, yo:yo + 448,:]
            
            one_hot = np.zeros(2)
            one_hot[sev] = 1

            return img_np, one_hot
        except Exception as e:
            print(bcolors.FAIL, e, bcolors.ENDC, file=sys.stderr)
            print(bcolors.FAIL, img, bcolors.ENDC, file=sys.stderr)
            return np.zeros([16, 448, 448, 3]), np.zeros(2)

def generate_masks(N, s=8, p1=0.5, input_size=(448, 448)):
    cell_size = np.ceil(np.array(input_size) / s)
    up_size = (s + 1) * cell_size

    grid = np.random.rand(N, s, s) < p1
    grid = grid.astype('float32')

    masks = np.empty((N, *input_size))

    for i in tqdm(range(N), desc='Generating masks'):
        # Random shifts
        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        # Linear upsampling and cropping
        masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                anti_aliasing=False)[x:x + input_size[0], y:y + input_size[1]]
    masks = masks.reshape(-1, *input_size, 1)
    return masks


checkpoint_path = "/work/jprieto/data/remote/EGower/jprieto/train/train_stack_efficientv2s_01042022_weights/train_stack_efficientv2s_01042022"

model = TTModel()
model.load_weights(checkpoint_path)
model.build(input_shape=(1, 16, 448, 448, 3))
model.summary()

model_pred = TTModelPred()
model_pred.features = model.features
model_pred.V = model.V
model_pred.P = model.P
model_pred.build(input_shape=(None, 448, 448, 3))
model_pred.summary()

# csv_path_stacks = "/work/jprieto/data/remote/EGower/jprieto/trachoma_bsl_mtss_besrat_field_mislabeled.csv"
# csv_path_stacks = "hinashah/hinashah_applist/jimma_tis_may_2022_discordant_NOTT.csv"
csv_path_stacks = "hinashah/hinashah_applist/jimma_tis_may_2022_discordant_NOTT_review.csv"

test_df = pd.read_csv(csv_path_stacks).replace("hinashah/", "", regex=True)
test_df['tt sev'] = (test_df['tt sev'] >= 1).astype(int)

dg_test = DatasetGenerator(test_df)

def test_generator():

    enqueuer = OrderedEnqueuer(dg_test, use_multiprocessing=True)
    enqueuer.start(workers=8, max_queue_size=128)

    datas = enqueuer.get()

    for idx in range(len(dg_test)):
        yield next(datas)

    enqueuer.stop()

dataset = tf.data.Dataset.from_generator(test_generator,
    output_signature=(tf.TensorSpec(shape = (None, 448, 448, 3), dtype = tf.float32), 
        tf.TensorSpec(shape = (2,), dtype = tf.int32))
    )

dataset = dataset.batch(1)
dataset = dataset.prefetch(16)

N = 2000
p1 = 0.5
s = 16 

out_masks = "/work/jprieto/data/remote/EGower/jprieto/masks_rise_explainability.npy"
if(os.path.exists(out_masks)):
    masks = np.load(out_masks)
else:
    masks = generate_masks(N, s=s, p1=p1)
    np.save(out_masks, masks)

for idx, (X, Y) in tqdm(enumerate(dataset), total=len(dg_test)):


    out_name = os.path.join("hinashah/hinashah_applist/jimma_tis_may_2022_discordant_NOTT_rise_explainability/", test_df.loc[idx]["image"].replace(".jpg", ".nrrd"))
    out_dir = os.path.dirname(out_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    if not os.path.exists(out_name):

        saliency_maps = []
        for x in X[0]:
            x = x*masks
            x = np.concatenate([model_pred.predict_on_batch(x[i:i+64]) for i in range(0, N, 64)])

            x = x.T.dot(masks.reshape(N, -1)).reshape(-1, 448, 448).transpose((1,2,0))
            x = x / N / p1
            saliency_maps.append(x)

        saliency_maps = sitk.GetImageFromArray(np.array(saliency_maps), isVector=True)

        writer = sitk.ImageFileWriter()
        writer.SetFileName(out_name)
        writer.UseCompressionOn()
        writer.Execute(saliency_maps)

    
    


    