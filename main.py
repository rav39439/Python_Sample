import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import base64
import glob
import io
from PIL import Image
from tensorflow import keras
from keras import layers
from flask import Flask, render_template, request, redirect, url_for,Response,jsonify

import numpy as np

from keras import preprocessing

from keras.preprocessing import image
from keras import Sequential
from keras.layers import Conv2D,Dense , MaxPool2D ,Flatten,Dropout,BatchNormalization

import pickle
# from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
from keras.preprocessing.image import ImageDataGenerator


batch_size = 16
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.4,
    horizontal_flip=True
)

#===================================Increasing training images=============================#
# img1='training/Bottle in _Not pressed_ state/*.jpeg'
# print(glob.glob(img1))
# for i in range(0 , len(glob.glob(img1))):
#         img =load_img(glob.glob(img1)[i])
#         x=img_to_array(img)
#         x=x.reshape((1,) + x.shape)
#         i=0
#         for batch in train_datagen.flow(x,batch_size=1,save_to_dir='preview/BottleNotpressed',save_prefix='bottleisnotpressed',save_format='jpeg'):
#             i+=1;
#             if(i>16):
#                  break
#
# img2='training/Bottle pressed state/*.jpeg'
# print(glob.glob(img2))
# for i in range(0 , len(glob.glob(img2))):
#         img =load_img(glob.glob(img2)[i])
#         x=img_to_array(img)
#         x=x.reshape((1,) + x.shape)
#         i=0
#         for batch in train_datagen.flow(x,batch_size=1,save_to_dir='preview/BottlePressed',save_prefix='bottleispressed',save_format='jpeg'):
#             i+=1;
#             if(i>4):
#                  break
# #=========================================================================================#
#
# #===============================Increasing Validation Images===============================#
# # img3='validation/BottleNotpressed/*.jpeg'
# # print(glob.glob(img3))
# # for i in range(0 , len(glob.glob(img3))):
# #         img =load_img(glob.glob(img3)[i])
# #         x=img_to_array(img)
# #         x=x.reshape((1,) + x.shape)
# #         i=0
# #         for batch in train_datagen.flow(x,batch_size=1,save_to_dir='validation/BottleNotpressed',save_prefix='bottleisnotpress',save_format='jpeg'):
# #             i+=1;
# #             if(i>4):
# #                  break
# # img4='validation/BottlePressed/*.jpeg'
# # print(glob.glob(img2))
# # for i in range(0 , len(glob.glob(img2))):
# #         img =load_img(glob.glob(img2)[i])
# #         x=img_to_array(img)
# #         x=x.reshape((1,) + x.shape)
# #         i=0
# #         for batch in train_datagen.flow(x,batch_size=1,save_to_dir='validation/BottlePressed',save_prefix='bottleispress',save_format='jpeg'):
# #             i+=1;
# #             if(i>2):
# #                  break
# #==============================================================================#
#
# #===============================model=======================================#
# train_ds = keras.utils.image_dataset_from_directory(
#     directory='preview/',
#     labels='inferred',
#     label_mode='int',
#     batch_size=16,
#     image_size=(256,256)
# )
#
# validation_ds = keras.utils.image_dataset_from_directory(
#     directory='validation/',
#     labels='inferred',
#     label_mode='int',
#     batch_size=16,
#     image_size=(256,256)
# )
# print(train_ds)
#
#
# model=Sequential()
#
# model.add(Conv2D(
#     32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(256,256,3)))
# model.add(MaxPool2D(pool_size=(2,2),strides=2,padding='valid'))
#
# model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2),strides=2,padding='valid'))
#
# model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2),strides=2,padding='valid'))
#
# model.add(Conv2D(256,kernel_size=(3,3),padding='valid',activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2),strides=2,padding='valid'))
#
# model.add(Flatten())
# model.add(Dense(128,activation='relu'))
# model.add(Dropout(0.05))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(1,activation='sigmoid'))
# model.summary()
# model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# history = model.fit(train_ds,epochs=10,validation_data=validation_ds)
# with open('modelupdated2.txt','wb') as f:
#     pickle.dump(model,f)
#===================================================================
#============================================================#

#====================model application======================#
my_string=""
def processimg(data):
 with open('modelupdated2.txt', 'rb') as f:
      mod=pickle.load(f)
      img=Image.open(io.BytesIO(data))
      im=img.resize((256,256))
      image = np.array(im)
      input_arr=np.array([image])
      classes = np.argmax(mod.predict(input_arr))
      print(mod.predict(input_arr)[0][0])
      if mod.predict(input_arr)[0][0]<0.5:
       return "Bottle is not pressed"
      else:
          return "Bottle is pressed"

app = Flask(__name__)


@app.route("/uploadFile", methods=['POST', "GET"])
def upload_image():
    if request.method == 'POST':
        result = ""
        data = request.files['file'].read()
        image = request.files['file']
        my_string = base64.b64encode(data)
        if image.filename == '':
           print("filename is invalid")
           resp={
            'message':"error",
           }
           return resp
        else:
           result = processimg(data)
           d = my_string.decode("utf-8")
           resp = Response(filename=d, result=result)
           return resp               
    else:
        resp={
               'message':"Method is not allowed",

        }

        return jsonify(resp)

            # return render_template('index.html', filename=d, result=result)
    # return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0",
            port=int(os.environ.get("PORT", 8080)))









# See PyCharm help at https://www.jetbrains.com/help/pycharm/
