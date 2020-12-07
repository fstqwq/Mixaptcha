from PIL import Image
import numpy as np
import random
import os
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import *

physical_device = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_device[0], True)

train_data_label = []

test_data_label = []

for home, dirs, files in os.walk(".\\test_data_0\\"):
    for filename in files:
        image_array = np.array(Image.open(home + filename))
        test_data_label.append([image_array, 0])

for home, dirs, files in os.walk(".\\test_data_1\\"):
    for filename in files:
        image_array = np.array(Image.open(home + filename))
        test_data_label.append([image_array, 1])

for home, dirs, files in os.walk(".\\train_data_0\\"):
    for filename in files:
        image_array = np.array(Image.open(home + filename))
        train_data_label.append([image_array, 0])

for home, dirs, files in os.walk(".\\train_data_1\\"):
    for filename in files:
        image_array = np.array(Image.open(home + filename))
        train_data_label.append([image_array, 1])


random.shuffle(test_data_label)
random.shuffle(train_data_label)


train_data = []
train_label = []
test_data = []
test_label = []

for i in test_data_label:
    test_data.append(i[0])
    test_label.append(i[1])
test_data_label = []
for i in train_data_label:
    train_data.append(i[0])
    train_label.append(i[1])
train_data_label = []

MAX_TRAIN_SIZE = 10000
MAX_TEST_SIZE = 1000

if len(train_data) > MAX_TRAIN_SIZE:
    train_data = train_data[:MAX_TRAIN_SIZE]
    train_label = train_label[:MAX_TRAIN_SIZE]
if len(test_data) > MAX_TEST_SIZE:
    test_data = test_data[:MAX_TEST_SIZE]
    test_label = test_label[:MAX_TEST_SIZE]

train_data = np.array(train_data).astype('float32')
train_label = np.array(train_label).astype('float32')
test_data = np.array(test_data).astype('float32')
test_label = np.array(test_label).astype('float32')


print(train_data.shape)
print(test_data.shape)

print("read data finished")


block_type = {18: 'basic block',
              34: 'basic block',
              50: 'bottlenect block',
              101: 'bottlenect block',
              152: 'bottlenect block'}

block_num = {18: (2, 2, 2, 2),
             34: (3, 4, 6, 3),
             50: (3, 4, 6, 3),
             101: (3, 4, 23, 3),
             152: (3, 4, 36, 3)}

filter_num = (64, 128, 256, 512)


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides=(1, 1), **kwargs):
        self.strides = strides
        if strides != (1, 1):
            self.shortcut = Conv2D(filters, (1, 1), name='projection', padding='same', use_bias=False)

        self.conv_0 = Conv2D(filters, (3, 3), name='conv_0', strides=strides, padding='same', use_bias=False)
        self.conv_1 = Conv2D(filters, (3, 3), name='conv_1', padding='same', use_bias=False)
        self.bn_0 = BatchNormalization(name='bn_0', momentum=0.9, epsilon=1e-5)
        self.bn_1 = BatchNormalization(name='bn_1', momentum=0.9, epsilon=1e-5)
        super(BasicBlock, self).__init__(**kwargs)

    def call(self, inputs, training):
        net = self.bn_0(inputs, training=training)
        net = tf.nn.relu(net)

        if self.strides != (1, 1):
            shortcut = tf.nn.avg_pool2d(net, ksize=(2, 2), strides=(2, 2), padding='SAME')
            shortcut = self.shortcut(shortcut)
        else:
            shortcut = inputs

        net = self.conv_0(net)
        net = self.bn_1(net, training=training)
        net = tf.nn.relu(net)
        net = self.conv_1(net)

        output = net + shortcut
        return output


class BottleneckBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides=(1, 1), projection=False, **kwargs):
        self.strides = strides
        self.projection = projection
        if projection or strides != (1, 1):
            self.shortcut = Conv2D(filters * 4, (1, 1), name='projection', padding='same', use_bias=False)

        self.conv_0 = Conv2D(filters, (1, 1), name='conv_0', padding='same', use_bias=False)
        self.conv_1 = Conv2D(filters, (3, 3), name='conv_1', strides=strides, padding='same', use_bias=False)
        self.conv_2 = Conv2D(filters * 4, (1, 1), name='conv_2', padding='same', use_bias=False)
        self.bn_0 = BatchNormalization(name='bn_0', momentum=0.9, epsilon=1e-5)
        self.bn_1 = BatchNormalization(name='bn_1', momentum=0.9, epsilon=1e-5)
        self.bn_2 = BatchNormalization(name='bn_2', momentum=0.9, epsilon=1e-5)
        super(BottleneckBlock, self).__init__(**kwargs)

    def call(self, inputs, training):
        net = self.bn_0(inputs, training=training)
        net = tf.nn.relu(net)

        if self.projection:
                shortcut = self.shortcut(net)
        elif self.strides != (1, 1):
                shortcut = tf.nn.avg_pool2d(net, ksize=(2, 2), strides=(2, 2), padding='SAME')
                shortcut = self.shortcut(shortcut)
        else:
            shortcut = inputs

        net = self.conv_0(net)
        net = self.bn_1(net, training=training)
        net = tf.nn.relu(net)
        net = self.conv_1(net)
        net = self.bn_2(net, training=training)
        net = tf.nn.relu(net)
        net = self.conv_2(net)

        output = net + shortcut
        return output


class ResNet(tf.keras.models.Model):
    def __init__(self, layer_num, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        if block_type[layer_num] == 'basic block':
            self.block = BasicBlock
        else:
            self.block = BottleneckBlock

        self.conv0 = Conv2D(64, (7, 7), strides=(2, 2), name='conv0', padding='same', use_bias=False)

        self.block_collector = []
        for layer_index, (b, f) in enumerate(zip(block_num[layer_num], filter_num), start=1):
            if layer_index == 1:
                if block_type[layer_num] == 'basic block':
                    self.block_collector.append(self.block(f, name='conv1_0'))
                else:
                    self.block_collector.append(self.block(f, projection=True, name='conv1_0'))
            else:
                self.block_collector.append(self.block(f, strides=(2, 2), name='conv{}_0'.format(layer_index)))

            for block_index in range(1, b):
                self.block_collector.append(self.block(f, name='conv{}_{}'.format(layer_index, block_index)))

        self.bn = BatchNormalization(name='bn', momentum=0.9, epsilon=1e-5)
        self.global_average_pooling = GlobalAvgPool2D()
        self.fc = Dense(1, name='fully_connected', activation='sigmoid', use_bias=False)

    def call(self, inputs, training):
        net = self.conv0(inputs)
        print('input', inputs.shape)
        print('conv0', net.shape)
        net = tf.nn.max_pool2d(net, ksize=(3, 3), strides=(2, 2), padding='SAME')
        print('max-pooling', net.shape)

        for block in self.block_collector:
            net = block(net, training)
            print(block.name, net.shape)
        net = self.bn(net, training)
        net = tf.nn.relu(net)

        net = self.global_average_pooling(net)
        print('global average-pooling', net.shape)
        net = self.fc(net)
        print('fully connected', net.shape)
        return net


model = ResNet(152)
model.build(input_shape=(None, 224, 224, 3))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

checkpoint_path = "mix_015/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=10)
model.save_weights(checkpoint_path.format(epoch=0))

model.fit(train_data, train_label, epochs=100, validation_data=(test_data, test_label), batch_size=10, callbacks=[cp_callback])

test_loss, test_acc = model.evaluate(test_data,  test_label, verbose=2)
print('\nTest accuracy:', test_acc)

"""
model = keras.Sequential()
model.add(layers.Conv2D(16, (3, 3), input_shape=(224, 224, 3)))
model.add(layers.Activation('relu'))
model.add(layers.AveragePooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(16, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.AveragePooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(32, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.AveragePooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(32, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.AveragePooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.AveragePooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.AveragePooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(300))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.85))
model.add(layers.Dense(1))
model.add(layers.Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_data, train_label, epochs=20, validation_data=(test_data, test_label), batch_size=5)
"""