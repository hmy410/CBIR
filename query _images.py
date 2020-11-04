import os, torch, glob
import numpy as np
from torch.autograd import Variable
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import shutil
import time
import h5py
import matplotlib.pyplot as plt
import imageio
from skimage import io
# import extract_features

# len=np.size(y,1)
# for i in len:

# hf=h5py.File('./dataset.h5','r')
# n1=hf.get('feature')
# n1=np.array(n1)
# print(n1.shape)

def AP(doc_list,rel_list,rel_num):
    sum=0
    loc=1
    for i in doc_list:
        if i in rel_list:
            sum+=loc/(doc_list.index(i)+1)
            loc+=1
    AP=sum/loc
    return AP

def mAP(query_AP,num):
    sum_AP=0
    for i in query_AP:
        sum_AP+=i
    mAP=sum_AP/num
    return mAP


def query(data_feature):
    data_dir = './256_ObjectCategories'
    # features_dir = './256_ObjectCategories'
    query_list = []
    files_list=[]
    x = os.walk(data_dir)

    for path, d, filelist in x:
        d.sort()
        filelist.sort()

        for filename in filelist:
            file_glob = os.path.join(path, filename)
            # file_glob.replace('\\', '/')
            files_list.extend(glob.glob(file_glob))

    # query_img = []
    query_AP=[]

    x = os.walk(data_dir)
    for path, d, filelist in x:
        d.sort()
        filelist.sort()
        if len(filelist) > 0:
            rel_list = []
            query_name = os.path.join(path, filelist[0])
            rel_num=len(filelist)

            for i in range (1,rel_num):
                file_glob =os.path.join(path, filelist[i]) #rel_list为相关图片
                rel_list.extend(glob.glob(file_glob))

            print("检索图片："+query_name)


            hdr = io.imread(query_name)
            plt.imshow(hdr)
            plt.axis('off')
            plt.show()

            file_index = files_list.index(query_name)
            query_feature = data_feature[file_index]
            # query_img.extend({query_name,query_feature})
            query_list = []
            for data_f,file_name in zip(data_feature,files_list):
                dist=np.linalg.norm(query_feature - data_f)
                query_list.append({'file_name':file_name,'dist':dist})
            query_list.sort(key=lambda x:x['dist'],reverse=False) #doc_list为检索出的图片
            doc_list=[]
            for i in range (1,51): #返回50张图片
                print("第"+str(i)+"张："+query_list[i]['file_name']+'，dist='+str(query_list[i]['dist'])) #img_dist[0]是图片本身，dist=0
                doc_list.append(query_list[i]['file_name'])
                # hdr = io.imread(query_list[i]['file_name'])
                # plt.imshow(hdr)
                # plt.axis('off')
                # plt.show()

            query_AP.append(AP(doc_list,rel_list,rel_num))

    # query_feature.append(data_feature[index])
    #
    # query_feature = np.array(query_feature)
    query_mAP = mAP(query_AP, num=256)
    return query_mAP

if __name__ == "__main__":
    # use_gpu = torch.cuda.is_available()

    hf = h5py.File('./dataset.h5', 'r')
    data_feature = hf.get('feature')
    data_feature = np.array(data_feature).reshape(-1,4096)

    mAP=query(data_feature)
    print("mAP："+str(mAP))
