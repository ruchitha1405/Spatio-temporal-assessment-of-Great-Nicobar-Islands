import numpy as np
#import pandas as pd
import cv2
import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Input, Embedding, Activation, Flatten, Dense
from keras.layers import Conv2D,MaxPooling2D,Dropout,Conv2DTranspose,UpSampling2D, BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import mean_absolute_error as mae

print(">>> LIBRARIES IMPORTED <<<")


def runn(model_path,sp_data_path,sp_data_path2,datapath,outpath):
    tmp={}
    tmp["model"]=model_path
    tmp["outpath"]=outpath
    tmp["sp_data_path"]=sp_data_path
    tmp["sp_data_path2"]=sp_data_path2
    tmp["datapath"]=datapath

    model = keras.models.load_model(model_path)
    SEED = 53
    BATCH_SIZE=1
    img_dg=ImageDataGenerator(rescale=1/255.)
    xtest_img_g=img_dg.flow_from_directory(sp_data_path,shuffle=False,batch_size=BATCH_SIZE,class_mode=None,target_size=(740,900))
    ytest_img_g=img_dg.flow_from_directory(sp_data_path2,shuffle=False,batch_size=BATCH_SIZE,class_mode=None,target_size=(740,900))

    arr=[]
    for f in sorted(os.listdir(datapath)):
        arr.append(f)
    # print(arr)
    ct = len(arr)
    j=0
    val=0.0
    while j<ct:
        m=BATCH_SIZE
        i=0
        a = xtest_img_g.next()
        b = ytest_img_g.next()
        while i<m and j<ct:
            test_img = a[i]

            grth=b[i]
            # print(grth)
            test_img_input=np.expand_dims(test_img, 0)
            pre = model.predict(test_img_input)[0]  
            maee=mae(grth.flatten(),pre.flatten())
            val+=maee
            pk=pre
            pre=pre*255.
            # print(pre.shape)
            pre=cv2.cvtColor(pre, cv2.COLOR_BGR2RGB)
            # print(pre.shape)
            # print(pre)
            # print(arr[j])
            write_path=outpath+arr[j]
            cv2.imwrite(write_path,pre)
            # out_write = "output/"+arr[j]
            # print(out_write)
            # cv2.imwrite(out_write,pre)

            # print(write_path)
            print(j+1,maee)
            i+=1
            j+=1  
    tmp["MAE"]=val/ct
    print("Average mae: ",val/ct)
    return tmp

dicc={}
mp="./newModels/"
MODELS=["final_.hdf5"]# "MM-073-0.013810.hdf5","MM-079-0.013382.hdf5","MM-081-0.012579.hdf5","MM-095-0.012671.hdf5","MM-099-0.012497.hdf5",
others=[["../data/rgb_images","../data/rgb_masks","../data/rgb_images/cloudy","/rgb/my_test_out/"]]
#,["./data2/train_images","./data2/train_masks","./data2/train_images/train","/train_out/"],["./data2/val_images","./data2/val_masks","./data2/val_images/val","/val_out/"]


for path in MODELS:
    model_path=mp+path
    for x in others:
        sp_data_path,sp_data_path2,datapath,outpath=x
        op=outpath
        outpath="./my_testing/"+path+outpath
        idx="".join([path,op])
        print(idx)
        dicc[idx]=runn(model_path,sp_data_path,sp_data_path2,datapath,outpath)
	
print(dicc)
with open("./my_testing/files/nicobar.json", "w") as outfile:
    json.dump(dicc, outfile)


