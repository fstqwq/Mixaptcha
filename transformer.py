from PIL import Image
import numpy as np
import random
import os

const_h = 224
const_w = 224
too_small_img_counter = 0
data_img_counter = 0

data_path = ".\\data\\"

li = []


def origin_to_data(file):
    global too_small_img_counter, data_img_counter, li

    print("dealing with " + file)
    im = Image.open(file)
    image_array = np.array(im)
    h = image_array.shape[0]
    w = image_array.shape[1]
    if w < const_w or const_h < const_h or len(image_array.shape) < 3 or image_array.shape[2] == 4:
        too_small_img_counter += 1
        print(str(too_small_img_counter) + "images too small")
        return
    new_image_array = None
    if w < h:
        new_image_array = image_array[(h - w) // 2:(h - w) // 2 + w, 0:w]
    else:
        new_image_array = image_array[0:h, (w-h)//2:(w-h)//2+h]
    nim = Image.fromarray(new_image_array)
    nim = nim.resize((const_h, const_w), Image.ANTIALIAS)
    data_img_counter += 1
    li.append(nim)
    #nim.save(data_path + str(data_img_counter) + '.jpg')


def trans_all(dir):
    for home, dirs, files in os.walk(dir):
        for filename in files:
            fullname = os.path.join(home, filename)
            origin_to_data(fullname)


trans_all(r'.\imageNet')

random.shuffle(li)

for i in range(len(li)):
    li[i].save(data_path + str(i + 1) + '.jpg')