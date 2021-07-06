from __future__ import print_function
import keras
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as ktf
import numpy as np
import itertools
import seaborn as sns
import EffDenseNet
from sklearn.utils import class_weight
import ResDenseNet
import ResDenseEffNet
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,TensorBoard,ReduceLROnPlateau,EarlyStopping
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score,roc_curve,auc
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras.backend as K
from keras.losses import categorical_crossentropy
from Cosine_Decay_with_Warmup import WarmUpCosineDecayScheduler

nb_classes = 4  # 二分类
batch_size = 8
epoches = 200
img_channels = 3

#配置tkf.set_session的运算方式为GPU。占用40%的最大显存
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config = config)
ktf.set_session(session)

#加载数据集
input1 = np.load('./train/data/x_train.npy')
input2 = np.load('./train/data/x_train.npy')

train_label=np.load('./train/data/y_train.npy')

train_input1,val_input1,train_label1,val_label1 = train_test_split(input1, train_label, train_size=0.9, random_state=121)
train_input2,val_input2,train_label2,val_label2 = train_test_split(input2, train_label, train_size=0.9, random_state=121)


X_train1 = train_input1.astype('float32')/255.0
X_train2 = train_input2.astype('float32')/255.0

X_val1 = val_input1.astype('float32')/255.0
X_val2 = val_input2.astype('float32')/255.0

Y_train = keras.utils.to_categorical(train_label1, nb_classes)
Y_val = keras.utils.to_categorical(val_label1,nb_classes)

print('load data successfully!')
print('Begin model training...')

#------------------------------------------------------------------------------------
model = EffDenseNet.ensemble(input_shape1= (224,224,3),input_shape2=(224,224,3),nb_classes=4)

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])

model.summary()
model_name = "./save_models/623/Ensemble7.hdf5"
checkpoint1 = ModelCheckpoint(filepath=model_name,
                               monitor='val_acc',
                                verbose=1,
                                save_best_only=True,
                                mode = 'max')
checkpoint2 = EarlyStopping(monitor='val_acc',
                            verbose=1,
                            patience=5,
                            mode='max',
                            restore_best_weights=True)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.001),
                                monitor='val_loss',
                                cooldown=0,
                               patience=3,
                               min_lr=0)

# cosine_decay_with_warmup衰减学习率
# Total epochs to train.
sample_count = len(X_train1)
# Number of warmup epochs.
warmup_epoch = 10
# Training batch size, set small value here for demonstration purpose.
epochs = 25
# Base learning rate after warmup.
learning_rate_base = 0.0001

total_steps = int(epochs * sample_count / batch_size)

# Compute the number of warmup batches.
warmup_steps = int(warmup_epoch * sample_count / batch_size)
# Compute the number of warmup batches.
warmup_batches = warmup_epoch * sample_count / batch_size

# Create the Learning rate scheduler.
warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                            total_steps=total_steps,
                                            warmup_learning_rate=4e-06,
                                            warmup_steps=warmup_steps,
                                            hold_base_rate_steps=5,)

callbacks = [checkpoint1,checkpoint2,lr_reducer,TensorBoard(log_dir='logs/{}'.format(model_name))]

history = model.fit([X_train1,X_train2],Y_train,
                    batch_size=batch_size,
                    epochs=epoches,
                     #validation_split=0.1,
                    validation_data=([X_val1,X_val2],Y_val),
                    verbose=1,
                    callbacks=callbacks)
#
model.save('./save_models/623/model_Ensemble7.h5')
#
# plt.figure('metrics', figsize=(10, 10))#建画布1:画布的名字 figsize：画布大小
# plt.subplot(211)#221：将整个figure分成2行2列，共2个子图，这里子图是第一个位置
# plt.plot(history.history['loss'],color = 'r',label = 'Training loss')
# plt.plot(history.history['val_loss'],color = 'g',label = 'Validation loss')
# plt.xlabel('epoches', size=14)
# plt.ylabel('Loss', size=14)
# plt.legend(loc='best',shadow = True)
# plt.grid(True)
# # #画第二个子图，跟上面一样
# plt.subplot(212)
# plt.plot(history.history['acc'],color = 'r',label = 'Training acc')
# plt.plot(history.history['val_acc'],color = 'g',label = 'Validation acc')
# plt.legend(loc='best')
# plt.grid(True)
# plt.xlabel('epoches', size=14)
# plt.ylabel('Acc', size=14)
# plt.savefig('./res_pic/paper/EffDenseNet_2_pic.jpg')
# #模型评估
scores = model.evaluate([X_val1,X_val2],Y_val)
print("val loss="+str(scores[0]))
print("val accuracy="+str(scores[1]))

print('-----------confusion matrix--------------')
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./res_pic/623/con_Ensemble7.jpg')

y_true = Y_val.argmax(axis=1)
y_pred = model.predict([X_val1, X_val2])
y_pred = y_pred.argmax(axis=1)
con_mat = confusion_matrix(y_true, y_pred)
target_names = ['none','infection','ischaemia','both']
plot_confusion_matrix(con_mat, classes=target_names, normalize=True, title='Confusion matrix')

labels =[0,1,2,3]
print("[INFO] evaluating network...")
print(classification_report(Y_val.argmax(axis=1), y_pred,labels=labels,target_names=target_names,digits=3))
