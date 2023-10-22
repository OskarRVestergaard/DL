import tensorflow as tf
#from tensorflow import keras
#from keras.datasets import mnist
#from keras import backend as K
#from matplotlib import pyplot as plt


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