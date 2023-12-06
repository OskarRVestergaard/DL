from os import environ, path
from absl import logging as absl_logging
from IPython.display import clear_output
from keras.utils import to_categorical
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import ParameterSampler
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
import os
from pathlib import Path
from glob import glob
import numpy as np
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import top_k_accuracy_score

# Our stuff
from labels import labels
import background_model_functions as mf

class_labels = labels
data_folder_path = Path(os.getcwd() + r"/Dataset/FoodSeg103/Images")
models_path = Path(os.getcwd() + r"/Models")
batch_size = 8
image_size = 128
num_classes = 104


def __load_images_combined__():
    num_train_images = 4983
    num_val_images = 2135
    train_images_path = Path(data_folder_path, r"img_dir/train")
    train_ann_path = Path(data_folder_path, r"ann_dir/train")
    test_images_path = Path(data_folder_path, r"img_dir/test")
    test_ann_path = Path(data_folder_path, r"ann_dir/test")

    train_images_paths = sorted(os.listdir(train_images_path))
    train_ann_paths = sorted(os.listdir(train_ann_path))
    test_images_paths = sorted(os.listdir(test_images_path))
    test_ann_paths = sorted(os.listdir(test_ann_path))

    train_images = train_images_paths[:num_train_images]
    train_masks = train_ann_paths[:num_train_images]
    val_images = test_images_paths[:num_val_images]
    val_masks = test_ann_paths[:num_val_images]

    train_images = [str(Path(train_images_path, img)) for img in train_images]
    train_masks = [str(Path(train_ann_path, img)) for img in train_masks]
    val_images = [str(Path(test_images_path, img)) for img in val_images]
    val_masks = [str(Path(test_ann_path, img)) for img in val_masks]

    image_paths = sorted(train_images + val_images)
    mask_paths = sorted(train_masks + val_masks)

    return image_paths, mask_paths


image_paths, mask_paths = __load_images_combined__()
train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(
    image_paths, mask_paths, test_size=0.2
)


def load_model(model_folder_name):
    folder_load_path = os.path.join(models_path, Path(model_folder_name))
    his_load_path = os.path.join(folder_load_path, Path(r"history.npy"))
    model_load_path = os.path.join(folder_load_path, Path(r"model.keras"))
    history = np.load(his_load_path, allow_pickle="TRUE").item()
    model = keras.models.load_model(model_load_path)
    return (model, history)


def save_model(model, history):
    dt_string = datetime.now().strftime("%d%m%Y-%H:%M:%S")
    save_name = "model_" + dt_string
    folder_save_path = os.path.join(models_path, Path(save_name))
    os.mkdir(folder_save_path)
    his_save_path = os.path.join(folder_save_path, Path(r"history.npy"))
    model_save_path = os.path.join(folder_save_path, Path(r"model.keras"))
    np.save(his_save_path, history)
    model.save(model_save_path)


def train_new_model(
    epochs,
    useEarlyStopping=True,
    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
    optimizer2=keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.0001),
    data_augmentation_config=None,
):
    train_data_generator = mf.CustomDataGenerator(
        train_image_paths,
        train_mask_paths,
        batch_size,
        image_size,
        num_classes,
        data_augmentation_config,
    )
    val_data_generator = mf.CustomDataGenerator(
        val_image_paths, val_mask_paths, batch_size, image_size, num_classes, None
    )
    model = mf.DeeplabV3Plus(image_size, num_classes, False)
    model.compile(
        optimizer=optimizer2,
        loss=loss,
        metrics=[
            keras.metrics.TopKCategoricalAccuracy(k=5),
            keras.metrics.OneHotMeanIoU(num_classes, ignore_class=0),
            keras.metrics.CategoricalAccuracy(),
        ],
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )

    history = model.fit(
        train_data_generator,
        steps_per_epoch=len(train_image_paths) // batch_size,
        validation_data=val_data_generator,
        validation_steps=len(val_image_paths) // batch_size,
        epochs=epochs,
        callbacks=([early_stopping] if useEarlyStopping else []),
    )

    return model, history

def AlwaysBackground():
    input = keras.Input(shape=(image_size, image_size, 3))
    model = tf.keras.models.Sequential()
    model.add(input)

    def predictBackground(x):
        zeroes = np.zeros(
            shape=(1, image_size, image_size, 104)
        )
        zeroes[0,:,:,0] = 1.
        return tf.convert_to_tensor(zeroes)

    model.add(keras.layers.Lambda(predictBackground, output_shape=(1, image_size, image_size, 104)))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.0001),
        loss=keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[
            keras.metrics.TopKCategoricalAccuracy(k=5),
            keras.metrics.OneHotMeanIoU(num_classes, ignore_class=0),
            keras.metrics.CategoricalAccuracy(),
        ],
    )
    return model


def train_new_model_fine_tuning(
    epochs,
    useEarlyStopping=True,
    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
    optimizer2=keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.0001),
    data_augmentation_config=None,
    brightness_contrast_augmentation_config=None,
):
    train_data_generator = mf.CustomDataGenerator(
        train_image_paths,
        train_mask_paths,
        batch_size,
        image_size,
        num_classes,
        data_augmentation_config,
        brightness_contrast_augmentation_config,
    )

    val_data_generator = mf.CustomDataGenerator(
        val_image_paths, val_mask_paths, batch_size, image_size, num_classes, None
    )

    # BASE TRAINING
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )



    ####
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    resnet50.trainable = False
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = mf.DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = mf.convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = mf.convolution_block(x)
    x = mf.convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same", kernel_initializer=keras.initializers.HeNormal(), activation='softmax')(x)

    model = keras.Model(inputs=model_input, outputs=model_output)
    ####
    
    
    model.compile(
        optimizer=optimizer2,
        loss=loss,
        metrics=[
            keras.metrics.TopKCategoricalAccuracy(k=5),
            keras.metrics.OneHotMeanIoU(num_classes, ignore_class=0),
            keras.metrics.CategoricalAccuracy(),
        ],
    )

    history = model.fit(
        train_data_generator,
        steps_per_epoch=len(train_image_paths) // batch_size,
        validation_data=val_data_generator,
        validation_steps=len(val_image_paths) // batch_size,
        epochs=200,
        callbacks=([early_stopping] if useEarlyStopping else []),
    )

    # FINE TUNING
    resnet50.trainable = True
    
    early_stopping_tuning = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.00001, weight_decay=0.0001),
        loss=keras.losses.CategoricalFocalCrossentropy(from_logits=False),
        metrics=[
            keras.metrics.TopKCategoricalAccuracy(k=5),
            keras.metrics.OneHotMeanIoU(num_classes, ignore_class=0),
            keras.metrics.CategoricalAccuracy(),
        ],
    )

    history_fine_tuned = model.fit(
        train_data_generator,
        steps_per_epoch=len(train_image_paths) // batch_size,
        validation_data=val_data_generator,
        validation_steps=len(val_image_paths) // batch_size,
        epochs=200,
        callbacks=([early_stopping_tuning] if useEarlyStopping else []),
    )
     
    # RETURN 
    return model, history, history_fine_tuned

def create_data_augmentation_config():
    # Find ud af at tilføj sandsynligheder til augmentations
    return lambda seed: (
        tf.keras.Sequential(
            [
                keras.layers.RandomFlip("horizontal", seed),
                keras.layers.RandomRotation(
                    0.15, fill_mode="constant", seed=seed, fill_value=0.0
                ),
                keras.layers.RandomZoom(height_factor=(0.2, -0.2), fill_mode='constant', fill_value=0.0, interpolation='bilinear', seed=seed),
            ]
        )
    )

def create_data_augmentation_config_crop():
    # Find ud af at tilføj sandsynligheder til augmentations
    return lambda seed: (
        tf.keras.Sequential(
            [
                keras.layers.RandomFlip("horizontal", seed),
                keras.layers.RandomRotation(
                    0.15, fill_mode="constant", seed=seed, fill_value=0.0
                ),
                keras.layers.RandomCrop(height=128, width=128, seed=seed),
            ]
        )
    )
def create_brightness_contrast_augmentation_config():
    return lambda seed: (
        tf.keras.Sequential(
            [
                keras.layers.RandomBrightness(factor=0.2, seed=seed),
                keras.layers.RandomContrast(factor=0.1, seed=seed),
            ]
        )
    )


def display_image(index, data_augmentation_config):  # Consider changing index to path
    a_seed = random.randint(0, 2000000000)
    augment = data_augmentation_config(a_seed)

    raw_img = tf.io.read_file(image_paths[index])
    img = tf.image.decode_png(raw_img, channels=3)
    if data_augmentation_config:
        img = augment(img)
    img.set_shape([None, None, 3])
    resized_img = tf.image.resize(images=img, size=[image_size, image_size])
    resized_image_for_display = np.ceil(resized_img) / 256
    plt.imshow(resized_image_for_display)
    plt.show()


def display_prediction(model, index):  # Consider changing index to path
    raw_img = tf.io.read_file(val_image_paths[index])
    raw_img = tf.image.decode_png(raw_img, channels=3)
    raw_img.set_shape([None, None, 3])
    resized_img = tf.image.resize(images=raw_img, size=[image_size, image_size])
    img_to_predict = tf.keras.applications.resnet50.preprocess_input(resized_img)[
        None, :, :, :
    ]  # Model is made to predict many images, only 1 means add None
    prediction = model(img_to_predict)
    print(prediction.shape)
    max_prediction = np.argmax(prediction, axis=-1)[0]
    print(max_prediction.shape)

    raw_mask = tf.io.read_file(val_mask_paths[index])
    raw_mask = tf.image.decode_png(raw_mask, channels=1)
    raw_mask.set_shape([None, None, 1])
    resized_mask = tf.image.resize(
        images=raw_mask, size=[image_size, image_size], method="nearest"
    )

    # Plotting
    fig = plt.figure(figsize=(12, 5))
    cm = plt.get_cmap("viridis", lut=num_classes)

    resized_image_for_display = np.ceil(resized_img) / 256
    fig.add_subplot(1, 3, 1)
    plt.imshow(resized_image_for_display)
    plt.axis("off")
    plt.title("Original (resized) image")

    resized_mask_for_display = np.squeeze(resized_mask)
    fig.add_subplot(1, 3, 2)
    plt.imshow(resized_mask_for_display, cm, vmin=0, vmax=num_classes)
    plt.axis("off")
    plt.title("True labels")

    fig.add_subplot(1, 3, 3)
    plt.imshow(max_prediction, cm, vmin=0, vmax=num_classes)
    plt.axis("off")
    plt.title("Predicted labels")

    fig.show()
