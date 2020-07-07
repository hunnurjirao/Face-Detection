import numpy as np
import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf 
from utils import *
from Face_model import *

folder = "dataset/img_folder"

df_train = pd.read_csv("dataset/train_csv.csv")

img_names_train=[]
img_labels_train=[]

img_names_train=[df_train['img_name'][i] for i in range(len(df_train))]
img_labels_train=[df_train['img_label'][i] for i in range(len(df_train))]

train_images = []
train_labels = []

for i,filename in enumerate(os.listdir(folder)):
    img = cv2.imread(os.path.join(folder,filename))


    img = cv2.resize(img,(256,256))
    filename = filename.split('.')[0]
    if filename in img_names_train:
        
        train_images.append(img)
        train_labels.append(img_labels_train[i])

train_images , train_labels=np.array(train_images) , np.array(train_labels)


train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=3, dtype='float32')

train_images=train_images/255.0
print("train images : " + str(train_images.shape))
print("train labels : " + str(train_labels.shape))

F_model = Face_model(input_shape=(256,256,3))

print("parameters : " + str(F_model.count_params()))

database = {}
database["Dileep"] = img_to_encoding("dataset/img_folder/dk3.jpeg", F_model)
database["Hunnur"] = img_to_encoding("dataset/img_folder/khr1.jpg", F_model)
database["Sai_Kiran"] = img_to_encoding("dataset/img_folder/sk3.jpeg", F_model)


F_model.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])

F_model.fit(train_images,train_labels,epochs=2)

predict("dataset/img_folder/khr8.jpg", "Sai_Kiran", database, F_model)

