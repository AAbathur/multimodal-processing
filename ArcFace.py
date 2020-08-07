import keras
import math
import h5py
import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.datasets import mnist
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


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
        ###将(图片向量，[0,0,1,0,1])的数据形式改为：(图片向量，[0,0,1,0,0])和(图片向量，[0,0,0,0,1])的形式 
        i += batch_size
        if i >= len_images: 
            i = 0
            flag += 1
        ###arcface 把图片向量和标签向量连起来作为输入
        x = np.concatenate((batch_images, batch_labels), axis=1)
        if mode != "test":
            print('\n flag= {}  x_shape: {} mode = {}'.format(flag, x.shape, mode))
        yield (x, batch_labels)
        #yield (new_images, new_labels)

#accuracy
def improved_accuracy(y_true, y_pred):
    print("y_true shape is {} and y_pred shape is {}".format(y_true.shape, y_pred.shape))
    # acc = |y_true AND y_pred| / |y_true OR y_pred|
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.50), 'float32')) #y_true AND y_pred
    y_pred = K.cast(K.greater(K.clip(y_pred,0,1), 0.50), 'float32')
    total = K.sum(y_true) + K.sum(y_pred) - true_positives #y_true OR y_pred
    acc = true_positives / total
    return acc

#precision
def Precision(y_true, y_pred):
    print("y_true shape is {} and y_pred shape is {}".format(y_true.shape, y_pred.shape))
    # precision = |y_true AND y_pred| / |y_pred|
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.50), 'float32'))
    pred_positives = K.sum(K.cast(K.greater(K.clip(y_pred, 0, 1), 0.50), 'float32'))
    precision = true_positives / (pred_positives + K.epsilon())
    return precision

#recall
def Recall(y_true, y_pred):
    print("y_true shape is {} and y_pred shape is {}".format(y_true.shape, y_pred.shape))
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

class ArcFaceLayer(keras.layers.Layer):
    def __init__(self,num_class,s=256,m=0.0001):
        super(ArcFaceLayer, self).__init__()
        self.s = s
        self.m = m
        self.num_class = num_class 
    def build(self,input_shape):
        self.kernal = self.add_weight(shape=(input_shape[-1]-self.num_class, self.num_class),initializer="glorot_normal",trainable=True)
        super().build(input_shape)
    def call(self,input):
        print("layer input is {}".format(input.shape))
        x = input[...,:int(-1*self.num_class)]
        y = input[...,int(-1*self.num_class):]
        x = K.l2_normalize(x, axis=-1)
        weights = K.l2_normalize(self.kernal, axis=0)
        cos_m = tf.cos(self.m)
        sin_m = tf.sin(self.m)
        threshold = tf.cos(math.pi - self.m)
        cos_t = tf.matmul(x, weights)
        cos_t2 = tf.square(cos_t)
        sin_t2 = 1.0 - cos_t2
        sin_t = tf.sqrt(sin_t2)
        cos_mt = self.s * (cos_t * cos_m - sin_t * sin_m)
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v), dtype="bool")
        keep_val = self.s * (cos_t - self.m * sin_m)
        cos_mt_tmp = tf.where(cond, cos_mt, keep_val)
        mask = y
        inv_mask = 1. - mask
        s_cos_t = self.s * cos_t
        output = s_cos_t * inv_mask + cos_mt_tmp * mask
        return output
class ArcFaceModel(keras.Model):
    def __init__(self, drop_out, batch_size, num_class):
        super(ArcFaceModel,self).__init__()
        self.drop_out = drop_out
        self.batch_size = batch_size
        self.num_class = num_class
        self.fc1 = keras.layers.Dense(2048, activation="tanh")
        self.bn = keras.layers.BatchNormalization()
        self.do = keras.layers.Dropout(rate=self.drop_out)
        self.fc2 = keras.layers.Dense(2048, activation="tanh")
        self.my_layer = ArcFaceLayer(num_class=self.num_class)
        self.sigmoid = keras.layers.Activation("sigmoid")
    def call(self,input):
        x = input[...,:int(-1*self.num_class)]
        y = input[...,int(-1*self.num_class):]
        x = self.fc1(x)
        x = self.bn(x)
        x = self.do(x)
        x = self.fc2(x)
        x = self.bn(x)
        x = self.do(x)
        x_y = tf.concat([x,y], axis=1)
        x = self.my_layer(x_y)
        x = self.sigmoid(x)
        return x
def Model_func(Batch_size, Drop_out, Num_class):
    #两层全连接模型
    model = ArcFaceModel(drop_out=Drop_out,batch_size=Batch_size,num_class=Num_class)
    model.build(input_shape=(None,2048+Num_class))
    model.summary()
    ##改loss此处修改loss函数
    model.compile(optimizer=keras.optimizers.Adagrad(), loss="binary_crossentropy", metrics=["binary_accuracy","Recall","Precision"])
    return model

def train(model_path):
    #训练模型
    NUM_EPOCHS = 20
    batch_size = 32
    drop_rate = 0.5
    num_class = 2688
    NUM_TRAIN_IMAGES = 198839 # int(331399 * 0.6)
    NUM_VAL_IMAGES = 66280    # int(331399 * 0.2)
    NUM_TEST_IMAGES = 66280   # int(331399 * 0.2)

    model = Model_func(batch_size, drop_rate, num_class)
    
    train_gen = datagen(batch_size, mode = 'train')
    val_gen = datagen(batch_size, mode = 'val')
    test_gen = datagen(batch_size, mode="test")

    H = model.fit(train_gen, steps_per_epoch=NUM_TRAIN_IMAGES//batch_size, validation_data=val_gen, 
                            validation_steps=NUM_VAL_IMAGES//batch_size-2, epochs=NUM_EPOCHS)

    print("model train done !!!!!!!")
    model.evaluate(test_gen,steps=(66280//32 + 1))
    #save model
    model.save_weights(model_path)
    # plot the training loss and accuracy
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0,NUM_EPOCHS), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0,NUM_EPOCHS), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0,NUM_EPOCHS), H.history["binary_accuracy"], label="train_acc")
    plt.plot(np.arange(0,NUM_EPOCHS), H.history["val_binary_accuracy"], label="val_acc")
    #修改
    plt.title("Training categorical_crossentropy Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    #修改
    plt.savefig(r"tmp\plot_ArcFace_806.png")

if __name__ == "__main__":
    model_path = r"model\test.h5"
    train(model_path)    
