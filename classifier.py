import numpy as np
import random
import os
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow_hub as hub
from tensorflow.keras.layers import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--model_save_path", type=str, required=True)

args = parser.parse_args()
dataset_path = args.dataset_path
model_save_path = args.model_save_path
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

physical_device = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_device[0], True)

def get_all_data(dataset_path):
    train_data_labels = [(1, 0)] * len(os.listdir(os.path.join(dataset_path, "train_data", "0"))) + [(0, 1)] * len(os.listdir(os.path.join(dataset_path, "train_data", "1")))

    train_data_paths = []
    train_data_paths += list(map(lambda file: os.path.join(dataset_path, "train_data", "0", file), os.listdir(os.path.join(dataset_path, "train_data", "0"))))
    train_data_paths += list(map(lambda file: os.path.join(dataset_path, "train_data", "1", file), os.listdir(os.path.join(dataset_path, "train_data", "1"))))
    
    test_data_labels = [(1, 0)] * len(os.listdir(os.path.join(dataset_path, "test_data", "0"))) + [(0, 1)] * len(os.listdir(os.path.join(dataset_path, "test_data", "1")))

    test_data_paths = []
    test_data_paths += list(map(lambda file: os.path.join(dataset_path, "test_data", "0", file), os.listdir(os.path.join(dataset_path, "test_data", "0"))))
    test_data_paths += list(map(lambda file: os.path.join(dataset_path, "test_data", "1", file), os.listdir(os.path.join(dataset_path, "test_data", "1"))))

    return train_data_paths, train_data_labels, test_data_paths, test_data_labels

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    # print(image)
    return image

def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label

train_data_paths, train_data_labels, test_data_paths, test_data_labels = get_all_data(dataset_path)

# load_and_preprocess_image(train_data_paths[0])

train_data_count = len(train_data_paths)
test_data_count = len(test_data_paths)

train_path_label_ds = tf.data.Dataset.from_tensor_slices((train_data_paths, train_data_labels))
test_path_label_ds = tf.data.Dataset.from_tensor_slices((test_data_paths, test_data_labels))

train_image_label_ds = train_path_label_ds.map(load_and_preprocess_from_path_label)
test_image_label_ds = test_path_label_ds.map(load_and_preprocess_from_path_label)

train_ds = train_image_label_ds.cache(filename='./cache.tf-data')
train_ds = train_ds.shuffle(buffer_size=train_data_count)
# train_ds = train_ds.repeat()
train_ds = train_ds.batch(16)
train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

test_ds = test_image_label_ds.shuffle(buffer_size=test_data_count)
# test_ds = test_ds.repeat()
test_ds = test_ds.batch(16)
test_ds = test_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# model = tf.keras.Sequential([
#     hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4",
#                trainable=True, arguments=dict(batch_norm_momentum=0.997)),
#     tf.keras.layers.Dense(1, activation="sigmoid"),
# ])

model = tf.keras.applications.ResNet50(
    include_top=True, weights=None, classes=2)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

checkpoint_path = os.path.join(model_save_path, "cp-{epoch:04d}.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=25)
model.save_weights(checkpoint_path.format(epoch=0))
model.fit(x=train_ds, epochs=200, validation_data=test_ds, callbacks=[cp_callback])

test_loss, test_acc = model.evaluate(test_ds, verbose=2)
print("test_acc", test_acc)