from os import environ, path
from absl import logging as absl_logging
from IPython.display import clear_output
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
import background_model_functions as mf
from sklearn.model_selection import train_test_split

class_labels={0:'background',1:'candy',2:'egg tart',3:'french fries',4:'chocolate',5:'biscuit',6:'popcorn',7:'pudding',8:'ice cream',9:'cheese butter',10:'cake',11:'wine',12:'milkshake',13:'coffee',14:'juice',15:'milk',16:'tea',17:'almond',18:'red beans',19:'cashew',20:'dried cranberries',21:'soy',22:'walnut',23:'peanut',24:'egg',25:'apple',26:'date',27:'apricot',28:'avocado',29:'banana',30:'strawberry',31:'cherry',32:'blueberry',33:'raspberry',34:'mango',35:'olives',36:'peach',37:'lemon',38:'pear',39:'fig',40:'pineapple',41:'grape',42:'kiwi',43:'melon',44:'orange',45:'watermelon',46:'steak',47:'pork',48:'chicken duck',49:'sausage',50:'fried meat',51:'lamb',52:'sauce',53:'crab',54:'fish',55:'shellfish',56:'shrimp',57:'soup',58:'bread',59:'corn',60:'hamburg',61:'pizza',62:' hanamaki baozi',63:'wonton dumplings',64:'pasta',65:'noodles',66:'rice',67:'pie',68:'tofu',69:'eggplant',70:'potato',71:'garlic',72:'cauliflower',73:'tomato',74:'kelp',75:'seaweed',76:'spring onion',77:'rape',78:'ginger',79:'okra',80:'lettuce',81:'pumpkin',82:'cucumber',83:'white radish',84:'carrot',85:'asparagus',86:'bamboo shoots',87:'broccoli',88:'celery stick',89:'cilantro mint',90:'snow peas',91:' cabbage',92:'bean sprouts',93:'onion',94:'pepper',95:'green beans',96:'French beans',97:'king oyster mushroom',98:'shiitake',99:'enoki mushroom',100:'oyster mushroom',101:'white button mushroom',102:'salad',103:'other ingredients'}
data_folder_path = Path(os.getcwd()+ r"/Dataset/FoodSeg103/Images")
models_path = Path(os.getcwd() + r"/Models")
batch_size = 8
image_size = 128
num_classes = 104
num_train_images = 4983
num_val_images = 2135

def __load_images_combined__():
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

    image_paths=sorted(train_images + val_images)
    mask_paths=sorted(train_masks + val_masks)

    return image_paths, mask_paths

image_paths, mask_paths = __load_images_combined__()
train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(image_paths, mask_paths, test_size=0.2)

def load_model(model_folder_name):
    folder_load_path = os.path.join(models_path, Path(model_folder_name))
    his_load_path = os.path.join(folder_load_path, Path(r"history.npy"))
    model_load_path = os.path.join(folder_load_path, Path(r"model.keras"))
    history=np.load(his_load_path,allow_pickle='TRUE').item()
    model = keras.models.load_model(model_load_path)
    return (model, history)

def save_model(model, history):
    dt_string = datetime.now().strftime("%d%m%Y-%H:%M:%S")
    save_name = "model_" + dt_string
    folder_save_path = os.path.join(models_path, Path(save_name))
    os.mkdir(folder_save_path)
    his_save_path = os.path.join(folder_save_path, Path(r"history.npy"))
    model_save_path = os.path.join(folder_save_path, Path(r"model.keras"))
    np.save(his_save_path,history)
    model.save(model_save_path)

def train_new_model(epochs, useEarlyStopping=True, loss=keras.losses.CategoricalCrossentropy(from_logits=False), optimizer2 = keras.optimizers.Adam(learning_rate = 0.001, weight_decay = 0.0001), data_augmentation_config=None):
    train_data_generator = mf.CustomDataGenerator(train_image_paths, train_mask_paths, batch_size, image_size, num_classes, data_augmentation_config)
    val_data_generator = mf.CustomDataGenerator(val_image_paths, val_mask_paths, batch_size, image_size, num_classes, None)
    model = mf.DeeplabV3Plus(image_size, num_classes)
    model.compile(
    optimizer=optimizer2,
      loss=loss,
      metrics=[
          keras.metrics.OneHotMeanIoU(num_classes, ignore_class=0),
          keras.metrics.CategoricalAccuracy(),
      ])

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    history = model.fit(
        train_data_generator,
        steps_per_epoch=num_train_images//batch_size,
        validation_data=val_data_generator,
        validation_steps=num_val_images//batch_size,
        epochs=epochs,
        callbacks=([early_stopping] if useEarlyStopping else [])
    )
    return model, history

def display_image(index):
    raw_img = tf.io.read_file(image_paths[index])
    img = tf.image.decode_png(raw_img, channels=3)
    plt.imshow(img)

def display_prediction(model, index):
    raw_img = tf.io.read_file(val_image_paths[index])
    raw_img = tf.image.decode_png(raw_img, channels=3)
    raw_img.set_shape([None, None, 3])
    resized_img = tf.image.resize(images=raw_img, size=[image_size, image_size])
    img_to_predict = tf.keras.applications.resnet50.preprocess_input(resized_img)[None,:,:,:] #Model is made to predict many images, only 1 means add None
    prediction = model(img_to_predict)
    max_prediction =  np.argmax(prediction, axis=-1)[0]

    raw_mask = tf.io.read_file(val_mask_paths[index])
    raw_mask = tf.image.decode_png(raw_mask, channels=1)
    raw_mask.set_shape([None, None, 1]) #Sanity check
    resized_mask = tf.image.resize(images=raw_mask, size=[image_size, image_size], method='nearest')

    #Plotting
    fig = plt.figure(figsize=(12, 5))
    cm = plt.get_cmap('viridis', lut=num_classes)

    resized_image_for_display = np.ceil(resized_img) / 256
    fig.add_subplot(1, 3, 1) 
    plt.imshow(resized_image_for_display) 
    plt.axis('off') 
    plt.title("Original (resized) image") 
    
    resized_mask_for_display = np.squeeze(resized_mask)
    fig.add_subplot(1, 3, 2) 
    plt.imshow(resized_mask_for_display, cm, vmin=0, vmax=num_classes) 
    plt.axis('off') 
    plt.title("True labels") 
    
    fig.add_subplot(1, 3, 3) 
    plt.imshow(max_prediction, cm, vmin=0, vmax=num_classes) 
    plt.axis('off') 
    plt.title("Predicted labels") 

    fig.show()
