from PIL import Image
import numpy as np
import random
import os

const_h = 224
const_w = 224
data_img_counter = 0

data_path = ".\\data\\"
mixer_path = ".\\mixed_image\\"

os.system('mkdir ' + data_path)
os.system('mkdir ' + mixer_path)
for i in ("0", "1"):
    for j in ("test_data", "train_data"):
        os.system("mkdir " + j + "_" + i)

def mixer(file1, file2):
    print("dealing with " + file1 + ", " + file2)
    image1 = Image.open(file1)
    image2 = Image.open(file2)

    image_array1 = np.array(image1)
    image_array2 = np.array(image2)

    new_image_array = (image_array1 * 0.5 + image_array2 * 0.5).astype('uint8')
    return Image.fromarray(new_image_array)


files_counter = 0

for home, dirs, files in os.walk(data_path):
    for filename in files:
        files_counter += 1

test_mixed_num = 3000
test_unmixed_num = 3000
train_mixed_num = 20000
train_unmixed_num = 20000

for i in range(1, test_mixed_num + 1):
    u = 1
    v = 1
    while u == v:
        u = random.randint(1, test_mixed_num)
        v = random.randint(1, test_mixed_num)
    mixed_image = mixer(data_path + str(u) + ".jpg", data_path + str(v) + ".jpg")
    mixed_image.save(".\\test_data_1\\" + str(i) + ".jpg")

for i in range(test_mixed_num + 1, test_unmixed_num + test_mixed_num + 1):
    mixed_image = Image.open(data_path + str(i) + ".jpg")
    mixed_image.save(".\\test_data_0\\" + str(i - test_mixed_num) + ".jpg")

for i in range(test_unmixed_num + test_mixed_num + 1, test_unmixed_num + test_mixed_num + train_mixed_num + 1):
    u = 1
    v = 1
    while u == v:
        u = random.randint(test_unmixed_num + test_mixed_num + 1, test_unmixed_num + test_mixed_num + train_mixed_num + 1)
        v = random.randint(test_unmixed_num + test_mixed_num + 1, test_unmixed_num + test_mixed_num + train_mixed_num + 1)
    mixed_image = mixer(data_path + str(u) + ".jpg", data_path + str(v) + ".jpg")
    mixed_image.save(".\\train_data_1\\" + str(i - test_unmixed_num - test_mixed_num) + ".jpg")

for i in range(test_unmixed_num + test_mixed_num + train_mixed_num + 1, test_unmixed_num + test_mixed_num + train_mixed_num + train_unmixed_num + 1):
    mixed_image = Image.open(data_path + str(i) + ".jpg")
    mixed_image.save(".\\train_data_0\\" + str(i - test_unmixed_num - test_mixed_num - train_mixed_num) + ".jpg")

