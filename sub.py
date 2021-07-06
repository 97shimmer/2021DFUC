import keras
import cv2 as cv
import EffDenseNet
import ResDenseNet
import ResDenseEffNet
import numpy as np
import csv
import tensorflow as tf
from keras import backend as K
from keras.preprocessing import image
from keras.models import Model
from PIL import Image
import os

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config = config)
K.set_session(session)

csvFile = open("test_submit1.csv", "w")  # 创建csv文件
writer = csv.writer(csvFile)  # 创建写的对象
# 先写入columns_name
writer.writerow(["image",  "none", "infection", "ischaemia", "both"])  # 写入列的名称

test_root = './test/'
img_test = os.listdir(test_root)

model = EffDenseNet.ensemble(input_shape1=(224, 224, 3), input_shape2=(224, 224, 3), nb_classes=4)
model.load_weights('/home/new/cyq/ResNet_test/demo1/two_channel_test/2021DFU/save_models/623/Ensemble_2.h5')
# model.load_weights('/home/new/cyq/ResNet_test/demo1/two_channel_test/2021DFU/save_models/paper/model_EffDenseNet_5fold.h5')
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])
#-------------------------------------------------------------------------------------------------
for i in range(len(img_test)):
    image = cv.imread(test_root + img_test[i]).astype("float32") / 255.0
    x1 = np.expand_dims(image, axis=0)
    x2 = np.expand_dims(image, axis=0)
    x3 = np.expand_dims(image, axis=0)
    preds = model.predict([x1, x2])
    print(preds)
    writer.writerow([img_test[i],preds[0][0],preds[0][1],preds[0][2],preds[0][3]])
csvFile.close()


