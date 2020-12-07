from PIL import Image
import numpy as np
import random
import os

const_h = 224
const_w = 224
data_img_counter = 0

data_path = ".\\data\\"

for i in ("0", "1"):
    for j in ("test_data", "train_data"):
        for k in ("0", "1", "2"):
            os.system("mkdir " + j + "_" + k + "_label_" + i)

def mixer(file1, file2):
    print("dealing with " + file1 + ", " + file2)
    image1 = Image.open(file1)
    image2 = Image.open(file2)

    image_array1 = np.array(image1)
    image_array2 = np.array(image2)

    new_image_array = (image_array1 * 0.3 + image_array2 * 0.7).astype('uint8')
    return Image.fromarray(new_image_array)


files_counter = 0

for home, dirs, files in os.walk(data_path):
    for filename in files:
        files_counter += 1


test_mixed_num = 1000
test_unmixed_num = 1000
train_mixed_num = 7000
train_unmixed_num = 7000

for dataset in ("0", "1", "2"):
    for i in range(1, test_mixed_num + 1):
        u = 1
        v = 1
        while u == v:
            u = random.randint(1, test_mixed_num + 1)
            v = random.randint(1, test_mixed_num + 1)
        u = u + int(dataset) * (test_mixed_num + test_unmixed_num + train_mixed_num + train_unmixed_num)
        v = v + int(dataset) * (test_mixed_num + test_unmixed_num + train_mixed_num + train_unmixed_num)
        mixed_image = mixer(data_path + str(u) + ".jpg", data_path + str(v) + ".jpg")
        mixed_image.save(".\\test_data_" + dataset + "_label_1\\" + str(i) + ".jpg")

    for i in range(test_mixed_num + 1, test_unmixed_num + test_mixed_num + 1):
        u = i + int(dataset) * (test_mixed_num + test_unmixed_num + train_mixed_num + train_unmixed_num)
        mixed_image = Image.open(data_path + str(u) + ".jpg")
        mixed_image.save(".\\test_data_" + dataset + "_label_0\\" + str(i - test_mixed_num) + ".jpg")

    for i in range(test_unmixed_num + test_mixed_num + 1, test_unmixed_num + test_mixed_num + train_mixed_num + 1):
        u = 1
        v = 1
        while u == v:
            u = random.randint(test_unmixed_num + test_mixed_num + 1, test_unmixed_num + test_mixed_num + train_mixed_num + 1)
            v = random.randint(test_unmixed_num + test_mixed_num + 1, test_unmixed_num + test_mixed_num + train_mixed_num + 1)
        u = u + int(dataset) * (test_mixed_num + test_unmixed_num + train_mixed_num + train_unmixed_num)
        v = v + int(dataset) * (test_mixed_num + test_unmixed_num + train_mixed_num + train_unmixed_num)
        mixed_image = mixer(data_path + str(u) + ".jpg", data_path + str(v) + ".jpg")
        mixed_image.save(".\\train_data_" + dataset + "_label_1\\" + str(i - test_unmixed_num - test_mixed_num) + ".jpg")

    for i in range(test_unmixed_num + test_mixed_num + train_mixed_num + 1, test_unmixed_num + test_mixed_num + train_mixed_num + train_unmixed_num + 1):
        u = i + int(dataset) * (test_mixed_num + test_unmixed_num + train_mixed_num + train_unmixed_num)
        mixed_image = Image.open(data_path + str(u) + ".jpg")
        mixed_image.save(".\\train_data_" + dataset + "_label_0\\" + str(i - test_unmixed_num - test_mixed_num - train_mixed_num) + ".jpg")
