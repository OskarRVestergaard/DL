import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras import backend as K
from matplotlib import pyplot as plt

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# dtype = torch.float
# device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.set_default_device(device)