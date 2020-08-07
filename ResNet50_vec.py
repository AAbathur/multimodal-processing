#codding: utf-8

""" 获得ResNet模型的倒数第二层 """

import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import time
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
import random
import csv
import glob
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

import numpy as np
import gc
from multiprocessing.pool import Pool
from multiprocessing import Process


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

vocab_path = r"data\vocab.txt"

def vocab_list(vocab_data=vocab_path) :
    #读取vocab_data到vocab_dic,并返回
    vocab_list = []
    with open(vocab_data,'r', encoding='utf-8') as f :
        for line in f.readlines():
            vocab_list.append(line.strip())  
    return vocab_list

def load_data(image_path, url_tag):
    """ 读取image_path的图片，从url_tag文件中获得每个图片对应的tag 
        返回图片路径list和对应的label list
    """
    vocab = vocab_list()
    images = []
    labels = []
    line_num_list = []
    with open(url_tag,'r',encoding='utf-8') as f1:
        all_line = f1.readlines()
        for line in all_line:
            line_list = line.strip().split('\t')
            line_num_list.append(line_list[0])
        count = 0
        for image_name in os.listdir(r"./"+image_path):
            count += 1
            print("count : {}".format(count))
            
            label_list = [0 for _ in range(len(vocab))]
            if image_name[-4:] != '.jpg':continue
            line_num = image_name[:-4]
            line = all_line[line_num_list.index(line_num)]
            line_list = line.strip().split('\t')
            assert int(line_list[0]) == int(line_num)
            
            for i in line_list[2:]:
                label_list[vocab.index(i)] = 1  
            images.append(image_path+'/'+image_name)
            labels.append(label_list)
    return (images,labels)
def data_path(image_path,url_tag,outpath):
    #将图片路径和label写入txt
    images,labels = load_data(image_path, url_tag)
    with open(outpath,'w',encoding="utf-8") as f:
        writer = csv.writer(f)
        assert len(images) == len(labels)
        for i in range(len(images)):
            print("i: {}".format(i))
            image = images[i]
            label = labels[i]
            writer.writerow([image] + label)
    print("write data path done!!!")
def count_imgs(images_path):
    count = 0
    for img in os.listdir(r'./'+ images_path):
        count += 1
    return count
def DenseNet121_vec(images):
    #images:list of image path
    base_model = keras.applications.DenseNet121(include_top=False,weights="imagenet")
    # DenseNet121:include_top=False: output shape:(7,7,1024)
    # DenseNet121:include_top=True: output shape:(1000) 
    # 最后一层做 7*7 global average pool + 1000维全连接+softmax
    test_imgs = []
    for img in images:
        img = image.load_img(img, target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = keras.applications.densenet.preprocess_input(x)
        # x.shape: (1,224,,224,3)
        test_imgs.append(x)
    test_imgs = np.array(test_imgs)
    test_data = tf.squeeze(test_imgs,axis=1)
    #test_data.shape: (200,224,224,3) 
    ###调用预训练模型，得到对应的向量
    preds = base_model.predict(test_data, steps = 1)
    del test_imgs
    del test_data
    gc.collect()
    return preds
def get_vec_by_ResNet50(images):  
    base_model = keras.applications.ResNet50(weights='imagenet')
    model = keras.models.Model(inputs = base_model.input, outputs = base_model.get_layer(index=175).output)
    test_imgs = []
    ### 读取图片路径，进行处理
    for img in images:
        img = image.load_img(img, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        test_imgs.append(x)
    test_imgs = np.array(test_imgs)
    test_data = tf.squeeze(test_imgs, axis=1) 
    ###调用预训练模型，得到对应的向量
    preds = model.predict(test_data, steps = 1)
    del test_imgs
    del test_data
    gc.collect()
    return preds
def read_images_txt(images_path, start_line, line_num):
    #读取image_path 文件路径中的图片路径, 起始点在start_line， 读取line_num行
    images = []
    with open(images_path, 'r', encoding='utf-8') as f1:
        if start_line+line_num > 331399:
            for line in f1.readlines()[start_line:]:
                line = line.strip()
                images.append(line)
        else:
            for line in f1.readlines()[start_line:start_line+line_num ]:
                line = line.strip()
                images.append(line)
    return images
def get_vec(count):
    #将10次循环封装起来
    txt_path = r"image-data\images.txt"
    batch_size = 100
    steps = 10
    start = count * 1000
    print("\n the start line of picture is : {}".format(start))
    images = read_images_txt(txt_path, start, batch_size * steps)
    X = np.zeros(shape=(len(images) , 2048))
    for i in range(steps):
        x_step = get_vec_by_ResNet50(images[i*batch_size:(i+1)*batch_size])
        X[i * batch_size: i* batch_size + x_step.shape[0]] = x_step
    np.savez(r"image-data\imgs\imgs_"+str(count+1) +".npz", X)

def get_vec_by_Dense121(count=0):
    txt_path = r"image-data\images.txt"
    bz = 100
    start = count * 10000
    images = read_images_txt(txt_path,start,10000)
    print("length of images is {}".format(len(images)))
    X = np.zeros(shape=(len(images),7,7,1024))
    for step in range(round(len(images)/bz)):
        step_images = images[step*bz:min((step+1)*bz,len(images))]
        pred_vec = DenseNet121_vec(step_images)
        print("step: {} pred_vec shape:{}".format(step,pred_vec.shape))
        X[step*bz:min((step+1)*bz,len(images))] = pred_vec
    np.savez(r"image-data\imgs\imgs_"+str(count+1) +".npz", X)
    print("get vec by dense121 done!")
def summary_npz():
    ###将前面分别处理的npz文件汇总到"image-data\images.npz"中
    a = np.zeros(shape=(331399,2048))
    i = 0
    for flag in range(1,333):
        file_name = r"image-data\imgs\imgs_"+str(flag)+".npz"
        print(file_name)
        x = np.load(file_name)
        a[i: i+x['arr_0'].shape[0]] = x['arr_0']
        i += x['arr_0'].shape[0]
    print(i)
    print(a[-1,:])
    np.savez(r"image-data\images.npz", a)
    print('all done') 
if __name__ == "__main__":
    images_path = r"image-data\images"
    url_path = r"image-data\url_tag_new.txt"
    #images, labels = load_data(images_path,url_path)
    #data_path(images_path,url_path,outpath=r"image-data\image_path_tag.csv")
    #get_vec_by_Dense121(count=0)
    """ count = 34
    npz_file = r"image-data\imgs\imgs_" + str(count) + r".npz"
    image_vec = np.load(npz_file)['arr_0']
    print(image_vec.shape) """
    
    """ all_labels = np.array(labels)
    print(all_labels.shape)
    np.savez(r'image-data\labels_new.npz',all_labels)
    print('savez done') """
    """ for j in range(25,28):
        p = Process(target=get_vec_by_Dense121, args=(j,))
        p.start()
        p.join() """
    
   




