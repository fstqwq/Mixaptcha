import shutil
import random
from PIL import Image
import numpy as np
from torchvision import transforms, utils
from torchvision import datasets
import torchvision.models as models
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
import torch.utils.data
import threading
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sema = threading.Semaphore(100)

train_size = 2500

for home, dirs, files in os.walk(r'.\dataset1'):
    for filename in files:
        fullname = os.path.join(home, filename)
        os.remove(fullname)

for home, dirs, files in os.walk(r'.\dataset2'):
    for filename in files:
        fullname = os.path.join(home, filename)
        os.remove(fullname)


def mixer(file1, file2):
    # print("dealing with " + file1 + ", " + file2)
    image1 = Image.open(file1)
    image2 = Image.open(file2)
    image_array1 = np.array(image1)
    image_array2 = np.array(image2)

    new_image_array = (image_array1 * 0.3 + image_array2 * 0.7).astype('uint8')
    return Image.fromarray(new_image_array)


def fill_one(path, name1, name2):
    with sema:
        mixer(name1, name2).save(os.path.join(os.path.join(path, r'mixed'), os.path.basename(name2)))


threads = []


def fill(path, names, times=1.0):
    file_cnt = 0
    for home, dirs, files in os.walk(os.path.join(path, r'mixed')):
        for filename in files:
            file_cnt += 1
    while file_cnt < times * train_size:
        file_cnt += 1
        name1 = names[len(names) - 1]
        name2 = names[len(names) - 2]
        new_thread = threading.Thread(target=fill_one, args=(path, name1, name2))
        threads.append(new_thread)
        new_thread.start()
        # fill_one(path, name1, name2)
        names.pop()
        names.pop()
    file_cnt = 0
    for home, dirs, files in os.walk(os.path.join(path, r'origin')):
        for filename in files:
            file_cnt += 1
    while file_cnt < times * train_size:
        file_cnt += 1
        name1 = names[len(names) - 1]
        shutil.copy(name1, os.path.join(path, r'origin'))
        names.pop()


bigdata = []

for home, dirs, files in os.walk(r'.\data_ACG'):
    for filename in files:
        fullname = os.path.join(home, filename)
        bigdata.append(fullname)

random.shuffle(bigdata)
fill(r'.\dataset1', bigdata, 1.0)
fill(r'.\dataset2', bigdata, 0.2)

for thread in threads:
    thread.join()


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


train_data = ImageFolderWithPaths(r'.\dataset1', transform=transforms.Compose([
    transforms.ToTensor()
]))

test_data = ImageFolderWithPaths(r'.\dataset2', transform=transforms.Compose([
    transforms.ToTensor()
]))

# print(train_data.classes)  # 获取标签
# print(test_data.classes)  # 获取标签

batch_size = 50

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

resnet = models.resnet50(num_classes=2).to(device)
optimizer = torch.optim.Adam(resnet.parameters())
loss_func = torch.nn.CrossEntropyLoss()

max_acc = 0.0
max_epoch = 20
max_eval_loss = 0.
last_epoch = max_epoch

aver = 0.0

for epoch in range(100):  # 训练的数据量为5个epoch，每个epoch为一个循环
    # 每个epoch要训练所有的图片，每训练完成200张便打印一下训练的效果（loss值）
    running_loss = 0.0  # 定义一个变量方便我们对loss进行输出

    resnet.train()

    cnt = 0

    for batch_x, batch_y, paths in train_loader:  # 这里我们遇到了第一步中出现的trailoader，代码传入数据
        cnt += 1
        # enumerate是python的内置函数，既获得索引也获得数据
        # get the inputs
        # wrap them in Variable
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)  # 转换数据格式用Variable

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()  # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度
        # forward + backward + optimize
        outputs = resnet(batch_x)  # 把数据输进网络
        loss = loss_func(outputs, batch_y)  # 计算损失值
        running_loss += loss.item() * len(batch_y) # loss累加
        loss.backward()  # loss反向传播
        optimizer.step()  # 反向传播后参数更新
        print('\r[epoch = %d, step = %4d/%4d]   loss: %.6f          ' % (epoch + 1, cnt, len(train_loader), running_loss / len(train_data)),  end='', flush=True)
    resnet.eval()
    eval_loss = 0.
    eval_acc = 0.
    with torch.no_grad():
        for batch_x, batch_y, paths in test_loader:
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            out = resnet(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.item() * len(batch_y)
            outputs = torch.max(out, 1)[1]
            num_correct = (outputs == batch_y).sum()
            eval_acc += num_correct.item()
    last_epoch -= 1
    if eval_acc > max_acc:
        last_epoch = max_epoch
        max_acc = eval_acc
        max_eval_loss = eval_loss
        torch.save(resnet, 'net_12.31.pkl')  # 保存整个神经网络的结构和模型参数
    print('Test Loss: {:.6f}, Acc: {:.6f}, Max_Acc: {:.6f}'.format(eval_loss / (len(test_data)), eval_acc / len(test_data), max_acc / (len(test_data))))

    if last_epoch == 0:
        break

# 保存神经网络
# torch.save(resnet.state_dict(), 'net_params.pkl')  # 只保存神经网络的模型参数

