import os, torch, glob
import numpy as np
from torch.autograd import Variable
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import time
import h5py



def save_h5(h5f, data, target):
    shape_list = list(data.shape)
    if not h5f.__contains__(target):
        shape_list[0] = None  # 设置数组的第一个维度是0
        dataset = h5f.create_dataset(target, data=data, maxshape=tuple(shape_list), chunks=True)
        return
    else:
        dataset = h5f[target]
    len_old = dataset.shape[0]
    len_new = len_old + data.shape[0]
    shape_list[0] = len_new
    dataset.resize(tuple(shape_list))  # 修改数组的第一个维度
    dataset[len_old:len_new] = data  # 存入新的文件


def extractor(img_path, net, use_gpu):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
    )
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    if use_gpu:
        x = x.cuda()
        net = net.cuda()
    y = net(x).cpu()
    y = y.data.numpy()
    return y[0]


if __name__ == '__main__':
    start=time.perf_counter()
    i = 0
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']

    files_list = []
    data_dir = './256_ObjectCategories'
    features_dir = './256_ObjectCategories'
    x = os.walk(data_dir)  # 读目录下的所有文件，用os.listdir，目录下还有目录，用os.walk

    hf = h5py.File('./dataset.h5')

    for path, d, filelist in x:
        d.sort()
        filelist.sort()

        for filename in filelist:
            file_glob = os.path.join(path, filename)
            # file_glob.replace('\\', '/')
            files_list.extend(glob.glob(file_glob))

    vgg16_feature_extractor = models.vgg16(pretrained=True)
    vgg16_feature_extractor.classifier = nn.Sequential(*list(vgg16_feature_extractor.classifier.children())[:-3])

    for param in vgg16_feature_extractor.parameters():
        param.requires_grad = False

    use_gpu = torch.cuda.is_available()

    for x_path in files_list:
        print(x_path)

        file_name = x_path.split('\\')[-1]
        y = extractor(file_name, vgg16_feature_extractor, use_gpu)
        save_h5(hf, data=y, target='feature')

        # print(y)
        i += 1 #统计一下提了多少张图片
        print(i)
    hf.close()
    elapsed=(time.perf_counter()-start)
    print("Time used:",elapsed)
