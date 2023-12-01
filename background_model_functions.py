from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils import Sequence
from keras.utils import to_categorical
import numpy as np
from PIL import Image
import random

##The model
def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same", kernel_initializer=keras.initializers.HeNormal(), activation='softmax')(x)


    # Adjust the number of output channels to match the number of classes
    # x_out = layers.Conv2D(104, (1, 1), activation='softmax')(model_output)

    # Use a Reshape layer to match the output shape to (height, width, num_classes)
    # x_out = layers.Reshape((image_size, image_size, 104))(x_out)


    return keras.Model(inputs=model_input, outputs=model_output)

class CustomDataGenerator(Sequence):
    def __init__(self, image_paths, mask_paths, batch_size, image_size, num_of_classes, data_augmentation_config=None, validation_split=0.2):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.image_size = image_size
        self.num_of_classes = num_of_classes
        self.use_augmentation = False
        if data_augmentation_config: #NOTE Nogle augments bør virke forskelligt for billede og labels (eksempelvis brightness)
            self.use_augmentation = True
            a_seed = random.randint(0, 2000000000)
            self.img_augment = data_augmentation_config(a_seed)
            self.mask_augment = data_augmentation_config(a_seed)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_x, batch_y = self.load_data(start, end)

        return batch_x, batch_y
        

    def load_data(self, start, end):
        images = []
        masks = []
        for i in range(start, min(end, len(self.image_paths))):            
            raw_img = tf.io.read_file(self.image_paths[i])
            img = tf.image.decode_png(raw_img, channels=3)
            if self.use_augmentation:
                img = self.img_augment(img)
            img.set_shape([None, None, 3])
            img = tf.image.resize(images=img, size=[self.image_size, self.image_size])
            img = tf.keras.applications.resnet50.preprocess_input(img)
            images.append(img)

            raw_img = tf.io.read_file(self.mask_paths[i])
            mask = tf.image.decode_png(raw_img, channels=1)
            if self.use_augmentation:
                mask = self.mask_augment(mask)
            mask.set_shape([None, None, 1])
            mask = tf.image.resize(images=mask, size=[self.image_size, self.image_size], method='nearest')
            one_hot_mask=to_categorical(mask,num_classes=self.num_of_classes)
            masks.append(one_hot_mask)
        return np.array(images), np.array(masks)