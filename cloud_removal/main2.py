import os
import cv2
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Embedding, Activation, Flatten, Dense
from keras.layers import Conv2D,MaxPooling2D,Dropout,Conv2DTranspose,UpSampling2D, BatchNormalization
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

print(">>> LIBRARIES IMPORTED <<<")

SEED = 53
BATCH_SIZE=8
EPOCHS=2

sp_data_path = "../input/clouddataset/data2/"
img_dg=ImageDataGenerator(rescale=1/255.)
xtrain_img_g=img_dg.flow_from_directory(sp_data_path+"train_images",seed=SEED,batch_size=BATCH_SIZE,class_mode=None,target_size=(740,900))
ytrain_img_g=img_dg.flow_from_directory(sp_data_path+"train_masks",seed=SEED,batch_size=BATCH_SIZE,class_mode=None,target_size=(740,900))
xval_img_g=img_dg.flow_from_directory(sp_data_path+"val_images",seed=SEED,batch_size=BATCH_SIZE,class_mode=None,target_size=(740,900))
yval_img_g=img_dg.flow_from_directory(sp_data_path+"val_masks",seed=SEED,batch_size=BATCH_SIZE,class_mode=None,target_size=(740,900))
train_generator = zip(xtrain_img_g,ytrain_img_g)
val_generator = zip(xval_img_g,yval_img_g)

print(">>> DATA GENERATOR DONE <<<")

def gen_model():
    input_layer = Input(shape=(740, 900, 3))  
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((4,4), padding='same')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((4,4))(x)
    output_layer = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(optimizer='adam' , loss='mean_squared_error', metrics=['mae'])
    return model


model = gen_model()
train_len=len(os.listdir(sp_data_path+"train_images/train"))
STEPS_PER_EPOCH = train_len//BATCH_SIZE
callback = EarlyStopping(monitor='loss', patience=5)
filepath="./models/MM-{epoch:03d}-{loss:.6f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='max')

print(">>> MODEL TRAINING STARTED <<<")

history=model.fit_generator(
    generator=train_generator,
    validation_data=val_generator,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=STEPS_PER_EPOCH,
    use_multiprocessing=True,
    callbacks=[callback,checkpoint],
    epochs=EPOCHS,
    verbose=1
)

print(">>> MODEL TRAINING DONE <<<")

model.save('./models/final_.hdf5')
with open("./files/history.json", "w") as outfile:
    json.dump(history.history, outfile)


print(">>> DUMP HISTORY <<<")
    
epoch_loss = history.history['loss']
epoch_val_loss = history.history['val_loss']
epoch_mae = history.history['mae']
epoch_val_mae = history.history['val_mae']

plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
plt.plot(range(0,len(epoch_loss)), epoch_loss, 'b-', linewidth=2, label='Train Loss')
plt.plot(range(0,len(epoch_val_loss)), epoch_val_loss, 'r-', linewidth=2, label='Val Loss')
plt.title('Evolution of loss on train & validation datasets over epochs')
plt.legend(loc='best')

plt.subplot(1,2,2)
plt.plot(range(0,len(epoch_mae)), epoch_mae, 'b-', linewidth=2, label='Train MAE')
plt.plot(range(0,len(epoch_val_mae)), epoch_val_mae, 'r-', linewidth=2,label='Val MAE')
plt.title('Evolution of MAE on train & validation datasets over epochs')
plt.legend(loc='best')

plt.savefig('./files/evolution.png')
print(">>> CHART SAVED <<<")