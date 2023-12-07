import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras import backend as K
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
# import tensorflow_datasets as tfds
from pathlib import Path

# currentPath = os.getcwd()
# img= keras.utils.load_img("Dataset/FoodSeg103/Images/ann_dir/test/00000048.png", target_size=(500,500))
# img1=tf.keras.utils.image_dataset_from_directory("Dataset/FoodSeg103/Images/ann_dir/test", labels="None")
# img2=tf.keras.utils.image_dataset_from_directory("/DataSet/FoodSeg/Images/ann_dir/test")
# train_data =  list(data_dir.glob('Dataset/FoodSeg103/*'))


devices = tf.config.list_physical_devices()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
with tf.device("/device:GPU:0"):
    # Create two random matrices

    a = tf.random.normal([1000, 1000])

    b = tf.random.normal([1000, 1000])

    # Multiply the matrices

    c = tf.matmul(a, b)

    print(c)
# dtype = torch.float
# device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.set_default_device(device)