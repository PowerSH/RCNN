import os
import cv2
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xml.etree.ElementTree as Et

from keras.layers import Dense
from keras import Model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16

from keras.optimizers import Adam
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import LinearSVR

from keras.callbacks import ModelCheckpoint, EarlyStopping

cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# 경로 지정(custom)
path = "JPEGImages_Sample/"
annot = "Annotations_Sample/"

def get_iou(box1, box2):
    x_lf = max(box1['x1'], box2['x1'])
    y_tp = max(box1['y1'], box2['y1'])
    x_rt = max(box1['x2'], box2['x2'])
    y_bt = max(box1['y2'], box2['y2'])

    area_intersection = (x_rt - x_lf) * (y_bt - y_tp)
    area_box1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    area_box2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])

    area_union = area_box1 + area_box2 - area_intersection

    iou = area_intersection / area_union
    return iou
'''
labels = [{'aeroplane': '1', 'bicycle': '2','bird': '3','boat': '4','bottle': '5',
           'bus': '6','car': '7', 'cat': '8','chair': '9','cow': '10',
           'diningtable': '11', 'dog': '12','horse': '13','motorbike': '14','person': '15',
           'pottedplant': '16', 'sheep': '17','sofa': '18','train': '19','tvmonitor': '20'}]
'''

train_images = []
train_labels = []

for cnt, i in enumerate(os.listdir(annot)):
    if cnt > 100: # 컴퓨터의 메모리를 감안해서 조절하면 됩니다.
        break
    if i.startswith("0"):
        filename = i.split(".")[0] + ".jpg"
        print(cnt, filename)

        image = cv2.imread(os.path.join(path,filename))
        tree = Et.parse(annot + i)
        root = tree.getroot()

        gtvalues = []
        for member in root.findall('object'):
            name = member.find('name').text
            x1 = int(member.find('bndbox/xmin').text)
            y1 = int(member.find('bndbox/ymin').text)
            x2 = int(member.find('bndbox/xmax').text)
            y2 = int(member.find('bndbox/ymax').text)
            gtvalues.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})

        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()
        ssresults = ss.process()
        imout = image.copy()

        temp = []
        counter = 0
        falsecounter = 0
        flag = 0
        fflag = 0
        bflag = 0
        for e, result in enumerate(ssresults):
            if e < 2000 and flag == 0:  # 이 부분도 조절은 됩니다. 2000이 최댓값입니다.
                for gtval in gtvalues:
                    x, y, w, h = result
                    iou = get_iou(gtval, {"x1": x, "x2": x+w, "y1": y, "y2": y+h})
                    if counter < 30:
                        if iou > 0.7:
                            timage = imout[y:y + h, x:x + w]
                            resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                            train_images.append(resized)
                            train_labels.append(root.find('object/name').text)
                            # train_labels.append(1)
                            counter += 1
                    else:
                        fflag = 1
                    if falsecounter < 30:
                        if iou < 0.3:
                            timage = imout[y:y + h, x:x + w]
                            resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                            train_images.append(resized)
                            train_labels.append(0)
                            falsecounter += 1
                    else:
                        bflag = 1
                if fflag == 1 and bflag == 1:
                    print("inside")
                    flag = 1

X_new = np.array(train_images)
y_new = np.array(train_labels)

vggmodel = VGG16(weights='imagenet', include_top=True)
vggmodel.summary()

for layers in (vggmodel.layers)[:15]:
    print(layers)
    layers.trainable = False

X = vggmodel.layers[-2].output

predictions = Dense(2, activation="softmax")(X)

model_final = Model(input=vggmodel.input, output=predictions)

opt = Adam(lr=0.0001)

model_final.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer = opt, metrics=['accuracy'])

model_final.summary()

enc = LabelBinarizer()
Y = enc.fit_transform(y_new)

X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.10)

trdata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
traindata = trdata.flow(x=X_train, y=y_train)
tsdata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
testdata = tsdata.flow(x=X_test, y=y_test)

checkpoint = ModelCheckpoint("VOC2007_VGG16_RCNN.h5", monitor='val_loss', verbose=1, save_best_only=True)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')

# epochs 도 조절은 됩니다만 최상의 결과를 위해서 그냥 두시는 걸 추천합니다.
# EarlyStopping 덕분에 1000번을 다 돌지는 않을 겁니다.
hist = model_final.fit_generator(generator=traindata, steps_per_epoch=10, epochs=1000, validation_data= testdata, validation_steps=2, callbacks=[checkpoint,early])


for cnt, i in enumerate(os.listdir(path)):
    if i.startswith("a"): # 테스트 셋의 첫 string입니다.
        img = cv2.imread(os.path.join(path, i))
        ss.setBaseImage(img)
        ss.switchToSelectiveSearchFast()
        ssresults = ss.process()
        imout = img.copy()
        imout = cv2.cvtColor(imout, cv2.COLOR_BGR2RGB)
        for e, result in enumerate(ssresults):
            if e < 2000:
                x, y, w, h = result
                timage = imout[y: y+h, x: x+w]
                resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                img = np.expand_dims(resized, axis=0)
                out = model_final.predict(img)
                if out[0][0] > 0.7:
                    cv2.rectangle(imout, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
        plt.imshow(imout)
        plt.show()
        break

print("end")