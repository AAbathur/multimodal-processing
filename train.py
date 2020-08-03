#coding:utf-8
""" 读取npz文件中的数据，进行训练 """
import h5py
import numpy as np
import time
import tensorflow as tf
import keras
from keras.layers import Dense,LSTM
import keras.backend as K
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
import xlsxwriter
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os
import math
import xlrd
import xlwt
import csv

""" os.environ['CUDA_DEVICE_ORDER'] = "PUS_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1' """

batch_size =32
epochs = 10
image_npz = r"image-data\images.npz"
label_npz = r"image-data\labels.npz"
vocab_path = r"data\vocab.txt"

def vocab_list(vocab_data=vocab_path) :
    #读取vocab_data到vocab_dic,并返回
    vocab_list = []
    with open(vocab_data,'r', encoding='utf-8') as f :
        for line in f.readlines():
            vocab_list.append(line.strip())
    print('the length of vocab list is{}'.format(len(vocab_list)))  
    return vocab_list

def load_X(mode):
    if mode == 'train':
        image_path = r"image-data\train_set\train_images.npz"
    elif mode == 'val':
        image_path = r"image-data\val_set\val_images.npz"
    elif mode == 'test':
        image_path = r"image-data\test_set\test_images.npz"
    X = np.load(image_path)['arr_0']
    print('load x, mode = {}'.format(mode))
    return X

def load_Y(mode):
    if mode == 'train':
        label_path = r"image-data\train_set\train_labels_new.npz"
        #label_path = r"image-data\train_set\train_labels.npz"
    elif mode == 'val':
        label_path = r"image-data\train_set\train_labels_new.npz"
        #label_path = r"image-data\val_set\val_labels.npz"
    elif mode == 'test':
        label_path = r"image-data\train_set\train_labels_new.npz"
        #label_path = r"image-data\test_set\test_labels.npz"
    Y = np.load(label_path)['arr_0']
    print('load y')
    return Y

def datagen(batch_size, mode='train'):
    images = load_X(mode)
    labels = load_Y(mode)
    len_images = images.shape[0]
    i = 0
    flag = 0
    while True:
        batch_images = images[i: min(i+batch_size,len_images),:]
        batch_labels = labels[i:min(i+batch_size, len_images),:]
        print('\n flag= {} i= {} , i/32 = {} batch_images: {} mode = {}'.format(flag, i, i//32, batch_images.shape, mode))
        ###将(图片向量，[0,0,1,0,1])的数据形式改为：(图片向量，[0,0,1,0,0])和(图片向量，[0,0,0,0,1])的形式 
        """ images_list = []
        labels_list = []
        for m in range(batch_images.shape[0]):
            count = int(np.sum(batch_labels[m]))
            image = np.expand_dims(batch_images[m], axis=0)
            one_image = np.repeat(image, count, axis=0)
            label = batch_labels[m]
            pos_idx = np.nonzero(label)[0]
            one_label = np.zeros(shape=(count,label.shape[0]))
            for j in range(count):
                one_label[j][pos_idx[j]] = 1.0
            images_list.append(one_image)
            labels_list.append(one_label)
        new_images = np.concatenate(images_list, axis=0)
        new_labels = np.concatenate(labels_list, axis=0)
        #shuffle array
        cur_state = np.random.get_state()
        np.random.shuffle(new_images)
        np.random.set_state(cur_state)
        np.random.shuffle(new_labels) """

        i += batch_size
        if i >= len_images: 
            i = 0
            flag += 1

        yield (batch_images, batch_labels)
        #yield (new_images, new_labels)

def multilabel_categorical_crossentropy(y_true, y_pred):
    """ 多标签分类的交叉熵, y_true和y_pred的shape相同,y_true的元素非0即1,1表示对应的类为目标类，0标语对应的类为非目标类 """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = K.zeros_like(y_pred[..., :1])
    y_pred_neg = K.concatenate([y_pred_neg, zeros], axis=-1)
    y_pred_pos = K.concatenate([y_pred_pos, zeros], axis=-1)
    neg_loss = K.logsumexp(y_pred_neg, axis=-1)
    pos_loss = K.logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss

def my_bce_loss(y_true, y_pred):
    """ mask掉在y_true中为0的部分，计算binary cross entropy """
    eps = K.epsilon()
    return -(y_true * K.log(y_pred+eps))
    #return -(y_true * K.log(y_pred+eps) + (1.0 - y_true) * K.log(1.0 - y_pred + eps))
    """ masked_y_pred = keras.layers.multiply(inputs=[y_true, y_pred])
    pos_loss = -K.sum( y_true * K.log(masked_y_pred + eps) + (1.0 - y_true) * K.log(1.0 - masked_y_pred + eps) , axis=1)
    pos_loss = pos_loss / K.sum(y_true, axis=1)

    neg_loss = K.sum((1.0 - y_true) * y_pred , axis=1)
    neg_loss = neg_loss / K.sum(1.0 - y_true ,axis=1)
    ## masked_pos 表示将y_pred中和y_true对应位置为1的置为1,其余位置值不变.
    # 例如 y_true = [0,0,1,1,0] y_pred =[0.1,0.2,0.3,0.4,0.5], masked_pos = [0.1,0.2,1,1,0.5]
    masked_pos = (1.0 - y_true) * y_pred + y_true 
    neg_loss = -K.sum(y_true * K.log(masked_pos + eps) + (1.0 - y_true) * K.log(1.0 - masked_pos + eps), axis=1)
    return 0.9 * pos_loss + 0.1 * neg_loss  """

def focal_loss(logits, labels, gamma):
    '''
    :param logits:  [batch_size, n_class]
    :param labels: [batch_size]
    :return: -(1-y)^r * log(y)
    '''
    softmax = tf.reshape(tf.nn.softmax(logits), [-1])  # [batch_size * n_class]
    labels = tf.range(0, logits.shape[0]) * logits.shape[1] + labels
    prob = tf.gather(softmax, labels)
    weight = tf.pow(tf.subtract(1., prob), gamma)
    loss = -tf.reduce_mean(tf.multiply(weight, tf.log(prob)))
    return loss 

#accuracy
def improved_accuarcy(y_true, y_pred):
    # acc = |y_true AND y_pred| / |y_true OR y_pred|
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.50), 'float32')) #y_true AND y_pred
    y_pred = K.cast(K.greater(K.clip(y_pred,0,1), 0.50), 'float32')
    total = K.sum(y_true) + K.sum(y_pred) - true_positives #y_true OR y_pred
    acc = true_positives / total
    return acc

#precision
def Precision(y_true, y_pred):
    # precision = |y_true AND y_pred| / |y_pred|
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.50), 'float32'))
    pred_positives = K.sum(K.cast(K.greater(K.clip(y_pred, 0, 1), 0.50), 'float32'))
    precision = true_positives / (pred_positives + K.epsilon())
    return precision

#recall
def Recall(y_true, y_pred):
    # recall = |y_true AND y_pred| / |y_true|
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.50), 'float32'))
    poss_positives = K.sum(K.cast(K.greater(K.clip(y_true, 0, 1), 0.50), 'float32'))
    recall = true_positives / (poss_positives + K.epsilon())
    return recall

#f-measure
def F_measure(y_true, y_pred):
    p_val = Precision(y_true, y_pred)
    r_val = Recall(y_true, y_pred)
    f_val = 2*p_val*r_val / (p_val + r_val)

    return f_val
def calculate_arcface_logits(x, weights, labels, class_num, s, m):
    x = K.l2_normalize(x, axis=1, name="normed_x")
    weights = K.l2_normalize(weights, axis=0)

    cos_m = K.cos(m)
    sin_m = K.sin(m)
    
    threshold = K.cos(math.pi - m)
    cos_t = K.dot(weights, x, name="cos_t")
    cos_t2 = K.square(cos_t, name="cos_t2")
    sin_t2 = K.subtract(1.0, cos_t2, name="sin_t2")
    sin_t = K.sqrt(sin_t2, name="sin_t")
    cos_mt = s * K.subtract(K.multiply(cos_t, cos_m), K.multiply(sin_t, sin_m), name="cos_mt")
    cond_v = cos_t - threshold
    cond = K.cast(K.relu(cond_v, name="condition"), dtype="bool")
    keep_val = s * (cos_t - m*sin_m)
    cos_mt_tmp = np.where(cond, cos_mt, keep_val)
    mask = labels
    inv_mask = K.subtract(1., mask, name="inverse_mask")
    s_cos_t = K.multiply(s, cos_t, name="scalar_cost_t")
    output = K.multiply(s_cos_t, inv_mask) + K.multiply(cos_mt_tmp, mask)

    return output
def Model_func(Batch_size, Drop_out):
    #两层全连接模型
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(2048,activation='tanh'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(rate=Drop_out))
    
    model.add(keras.layers.Dense(2048,activation='tanh'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(rate=Drop_out))

    #改loss此处修改激活函数

    model.add(keras.layers.Dense(2688, activation='sigmoid'))

    model.build(input_shape=(None,2048))
    model.summary()
    ##改loss此处修改loss函数
    model.compile(optimizer=keras.optimizers.Adagrad(), loss="binary_crossentropy", metrics=[improved_accuarcy,Precision,Recall])
    
    return model

def train(model_path):
    #训练模型
    NUM_EPOCHS = 40
    batch_size = 32
    drop_rate = 0.5
    NUM_TRAIN_IMAGES = 198839 # int(331399 * 0.6)
    NUM_VAL_IMAGES = 66280    # int(331399 * 0.2)
    NUM_TEST_IMAGES = 66280   # int(331399 * 0.2)

    model = Model_func(batch_size, drop_rate)
    train_gen = datagen(batch_size, mode = 'train')
    val_gen = datagen(batch_size, mode = 'val')

    H = model.fit(train_gen, steps_per_epoch=NUM_TRAIN_IMAGES//batch_size, validation_data=val_gen, 
                            validation_steps=NUM_VAL_IMAGES//batch_size-2, epochs=NUM_EPOCHS)
    #save model
    model.save(model_path)

    # plot the training loss and accuracy
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0,NUM_EPOCHS), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0,NUM_EPOCHS), H.history["val_loss"], label="val_loss")
    #plt.plot(np.arange(0,NUM_EPOCHS), H.history["binary_accuracy"], label="train_acc")
    #plt.plot(np.arange(0,NUM_EPOCHS), H.history["val_binary_accuracy"], label="val_acc")
    plt.plot(np.arange(0,NUM_EPOCHS), H.history["categorical_accuracy"], label="train_acc")
    plt.plot(np.arange(0,NUM_EPOCHS), H.history["val_categorical_accuracy"], label="val_acc")
    #修改
    plt.title("Training categorical_crossentropy Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    #修改
    plt.savefig(r"tmp\plot_multiclass_706.png")

def predict_func(model_path, pred_path):
    """ 由于内存问题，预测结果保存成文件后再做进一步处理 """
    batch_size = 32
    NUM_TEST_IMAGES = 66280   # int(331399 * 0.2)
    model = keras.models.load_model(model_path)
    test_gen = datagen(batch_size, mode = 'test')

    pred = model.predict(test_gen, steps=NUM_TEST_IMAGES//batch_size+1)
    print("predict done")
    print(pred.shape)
    np.savez(pred_path, pred)

def evaluate_func(model_path):
    """ 由于内存问题，预测结果保存成文件后再做进一步处理 """
    batch_size = 32
    NUM_TEST_IMAGES = 66280   # int(331399 * 0.2)
    model = keras.models.load_model(model_path)
    #val_gen = datagen(batch_size, mode = 'val')
    test_gen = datagen(batch_size, mode = 'test')

    model.evaluate(test_gen, steps=NUM_TEST_IMAGES//batch_size+1)
    print("evaluate done")
   

def pred2csv(y_true_path, pred_path, csv_path, image_txt):
    """ image_txt中是每个文件的路径，其中的每一行依次对应包含所有图片的npz文件(image-data\images.npz)中每一行2048的向量 """
    """ 根据方式，找到每个图片对应的真实标签和预测标签写入csv_path文件中 """
    vocab = vocab_list()
    idx_list = []
    shuffle_txt = r"image-data\shuffle_idx.txt"
    with open(shuffle_txt, 'r', encoding='utf-8') as f1:
        all_lines = f1.readlines()
        #for i in range(198839,265119):
        for i in range(265119,331399):
                idx = int(all_lines[i].strip())
                idx_list.append(idx)
    pred = np.load(pred_path)['arr_0']
    print('the shape of pred is {}'.format(pred.shape))
    y_true = np.load(y_true_path)['arr_0']
    print('the shape of y_true is {}'.format(y_true.shape))

    with open(image_txt, 'r', encoding='utf-8') as f2, open(csv_path,'w', encoding='utf-8') as f3:
        all_lines = f2.readlines()
        csv_writer = csv.writer(f3)
        total_recall = []
        total_precision = []
        for i,j in enumerate(idx_list):
            true_label = [vocab[m]  for m,n in enumerate(y_true[i].tolist()) if n==1 ]
            true_label_str = ' '.join(true_label)
            sort_idx = pred[i].argsort().tolist()
            pred_str = ""
            #count记录pred中预测正确的正类数量
            count = 0
            #用top_k指定取多少位作为预测结果
            top_k = 26
            for idx in reversed(sort_idx[-top_k:]):
                if vocab[idx] in true_label:
                    count += 1
                    pred_str += vocab[idx]+"|"+str(pred[i][idx])+"  "
                else:
                    pred_str += vocab[idx]+"  "
            if len(true_label):
                total_recall.append(count/len(true_label))
                total_precision.append(count/top_k)
            """ def mapfunc(a):
                if a > 0.5: return 1
                else : return 0
            vfunc = np.vectorize(mapfunc)
            #将原pred[i]中大于0.5的变为1,小于等于0.5的变为0
            pred_01 = vfunc(pred[i])
            pred_label = [vocab[m] for m,n in enumerate(pred_01) if n==1]
            pred_label_str = ' '.join(pred_label) """

            csv_writer.writerow([all_lines[j].strip(),'\n', true_label_str,'\n', pred_str])
    print("the length of total_recall list is: {}".format(len(total_recall)))
    print('mean recall is {}'.format(sum(total_recall)/len(total_recall))) 
    print('mean precision is {}'.format(sum(total_precision)/len(total_precision))) 
    print('pred2csv done')

def csv2xlsx(csv_file, xlsx_file):
    """ 预测结果写在csv_file中，将按照图片路径读取写入xlsx 文件 """
    workbook = xlsxwriter.Workbook(xlsx_file)
    worksheet = workbook.add_worksheet()

    with open(csv_file,'r', encoding='utf-8') as f1:
        reader = csv.reader(f1)
        count = 0
        for i, line in enumerate(reader):
            if i&1: continue
            image_path = line[0] 
            true_label = line[2]
            pred_label = line[4]
            print(i)
            worksheet.insert_image(i*10, 0, image_path, {'x_scale': 0.3, 'y_scale': 0.3})
            worksheet.write(i*10, 7, true_label)
            worksheet.write(i*10+2, 7, pred_label)
            if i>= 1800: break
        workbook.close()
    print('csv2xlsx done')

if __name__ == "__main__":
    ##修改
    #707 multi class
    model_path = r"model\model_multiclass_706.h5"
    pred_path = r"tmp\pred_multiclass_707.npz"
    csv_path = r"tmp\pred_top30_multiclass.csv"
    xlsx_path =r"tmp\pred_top_multiclass.xlsx"
  
    #train(model_path=model_path)
    predict_func(model_path=model_path, pred_path=pred_path)
    pred2csv(y_true_path=r"image-data\test_set\test_labels_new.npz", pred_path=pred_path,csv_path=csv_path, image_txt=r"image-data\images.txt")
    #evaluate_func(model_path) 
    #csv2xlsx(csv_path,xlsx_path)

    
    