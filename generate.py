from PIL import Image
import numpy as np
import random
import os
import shutil
import argparse

const_h = 224
const_w = 224

parser = argparse.ArgumentParser()
parser.add_argument("--train_size", type=int, default="2000")
parser.add_argument("--test_size", type=int, default="1000")
parser.add_argument("--source_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="dataset")
parser.add_argument("--mix_rate", type=float, default="0.5")

args = parser.parse_args()
source_dir = args.source_dir
output_dir = args.output_dir
train_size = args.train_size
test_size = args.test_size
mix_rate = args.mix_rate

def mixer(file1, file2):
    print("dealing with " + file1 + ", " + file2)
    image1 = Image.open(file1)
    image2 = Image.open(file2)

    image_array1 = np.array(image1)
    image_array2 = np.array(image2)

    new_image_array = (image_array1 * mix_rate + image_array2 * (1 - mix_rate)).astype('uint8')
    return Image.fromarray(new_image_array)

def save_image(image, id, dir):
    image.save(os.path.join(dir, str(id) + ".jpg"))

def copy_image(file, id, dir):
    shutil.copyfile(file, os.path.join(dir, str(id) + ".jpg"))

li = os.listdir(source_dir)
image_count = len(li)
print("source image count: ", image_count)
if train_size // 2 * 3 + test_size // 2 * 3 > image_count:
    print("required size is too large!")
    exit(0)
random.shuffle(li)
if not os.path.exists(os.path.join(output_dir, "train_data", "0")):
    os.makedirs(os.path.join(output_dir, "train_data", "0"))
if not os.path.exists(os.path.join(output_dir, "train_data", "1")):
    os.makedirs(os.path.join(output_dir, "train_data", "1"))
if not os.path.exists(os.path.join(output_dir, "test_data", "0")):
    os.makedirs(os.path.join(output_dir, "test_data", "0"))
if not os.path.exists(os.path.join(output_dir, "test_data", "1")):
    os.makedirs(os.path.join(output_dir, "test_data", "1"))

index = 0
for i in range(0, train_size // 2):
    save_image(
        mixer(os.path.join(source_dir, li[index]), os.path.join(source_dir, li[index + 1])), 
        i + 1, 
        os.path.join(output_dir, "train_data", "1"))
    index += 2
for i in range(train_size // 2, train_size):
    copy_image(
        os.path.join(source_dir, li[index]), 
        i - train_size // 2 + 1, 
        os.path.join(output_dir, "train_data", "0"))
    index += 1
for i in range(0, test_size // 2):
    save_image(
        mixer(os.path.join(source_dir, li[index]), os.path.join(source_dir, li[index + 1])), 
        i + 1, 
        os.path.join(output_dir, "test_data", "1"))
    index += 2
for i in range(test_size // 2, test_size):
    copy_image(
        os.path.join(source_dir, li[index]), 
        i - test_size // 2 + 1, 
        os.path.join(output_dir, "test_data", "0"))
    index += 1

