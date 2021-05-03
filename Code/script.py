import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Reshape, add
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras import regularizers
from skimage.util import random_noise
import cv2
import keras
import os
import pickle
from keras.models import model_from_json
import json
from PIL import Image, ImageFilter

print("CUDA_VISIBLE_DEVICES")
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

PATH=r'B:\Major\model'
os.listdir(PATH)

PATH2=r'B:\Major\datast'
os.listdir(PATH2)

PATH3=r'B:\Major\datast\output'

model = 'smudge_autoencoderV2.json'
weight = 'smudge_autoencoder_weightsV2.h5'

slice_size = 256
input_file_name = 'in4.png'
print("{}\{}".format(PATH2, input_file_name))

img = cv2.imread("{}\{}".format(PATH2, input_file_name), 0)
v_res = img.shape[0]
h_res = img.shape[1]
print(img.shape)

def apply_gamma(image):
    k=1.8
    image = 255 * (image/255)**k
    return image

img = apply_gamma(img)
print(img)

def count_blocks(res, slice_size):
  blocks = 0
  if res % slice_size == 0:
    blocks = int(res/slice_size)
  else:
    blocks = (int(res/slice_size)) + 1
  return blocks

def resizeImage(img, v_res, h_res, slice_size):
  v_res = count_blocks(v_res, slice_size) * slice_size
  h_res = count_blocks(h_res, slice_size) * slice_size
  img = cv2.resize(img, (h_res, v_res))
  print(v_res)
  print(h_res)
  return img

img = resizeImage(img, v_res, h_res, slice_size)
plt.imshow(img)
print(img.shape)

json_file = open(r'{}\{}'.format(PATH, model), 'r')
json_model = model_from_json(json_file.read())
json_model.load_weights(r'{}\{}'.format(PATH, weight))

# json_file.seek(0)
# json_file.read()

column = count_blocks(h_res, slice_size)
rows = count_blocks(v_res, slice_size)
print(column)
print(rows)

normalImg = img
newImg = None