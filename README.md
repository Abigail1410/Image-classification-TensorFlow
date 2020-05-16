# Image-classification-TensorFlow
Entrenamiento de red neuronal con layers con uso de Keras, Python y TensorFlow para la clasificación de imágenes Tigres y Pandas.
from google.colab import files
uploaded = files.upload()
Saving pandas-tigres (4).zip to pandas-tigres (4) (1).zip
In [0]:
import zipfile
import io
data = zipfile.ZipFile(io.BytesIO(uploaded['pandas-tigres (4).zip']), 'r')
data.extractall()
In [0]:
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
 
pandas_folder_path="/content/tigres"
tigres=[]
img_size=150
for img in os.listdir(pandas_folder_path):
    img = cv2.imread(os.path.join(pandas_folder_path,img))
    #img_gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_resize= cv2.resize(img,(img_size,img_size))
    tigres.append(img_resize)
In [0]:
tigres = np.array(tigres)
print(tigres.shape)
(6, 150, 150, 3)
In [0]:
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
 
pandas_folder_path="/content/pandas"
pandast=[]
img_size=150
for img in os.listdir(pandas_folder_path):
    img = cv2.imread(os.path.join(pandas_folder_path,img))
    #img_gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_resize= cv2.resize(img,(img_size,img_size))
    pandast.append(img_resize)
In [0]:
pandast = np.array(pandast)
print(pandast.shape)
(15, 150, 150, 3)
In [0]:
print(pandast[10].shape)
plt.figure()
plt.imshow(np.squeeze(pandast[10]))
plt.colorbar()
plt.grid(False)
plt.show()
(150, 150, 3)

In [0]:
plt.figure(figsize=(5,5))
for i in range(10):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(True)
    plt.imshow(Images[i])
    #, cmap=plt.cm.binary
    plt.xlabel(class_names[Labels[i]])
plt.show()
In [0]:
images = np.concatenate([tigres,pandast])
print(len(images))
Images = np.array(images)
print(Images.shape)
21
(21, 150, 150, 3)
In [0]:
etiquetas_tigres = np.repeat(0,6)
print(len(etiquetas_tigres))
print(etiquetas_tigres)
6
[0 0 0 0 0 0]
In [0]:
etiquetas_pandas = np.pandas(1,3)
print(len(etiquetas_pandas))
print(etiquetas_pandas)
In [0]:
class_names=['Tigre', 'Panda']
In [0]:
labels = np.concatenate([etiquetas_tigres,etiquetas_pandas])
print(len(labels))
print(labels)
Labels = np.array(labels)
print(Labels.shape)
In [0]:
plt.figure(figsize=(10,10))
for i in range(20):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(Images[i])
    #, cmap=plt.cm.binary
    plt.xlabel(class_names[Labels[i]])
plt.show()
In [0]:
from __future__ import absolute_import, division, print_function, unicode_literals

variable_name = ""
try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass

from __future__ import absolute_import, division, print_function, unicode_literals


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import tensorflow.keras.optimizers as Optimizer

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)
2.2.0
In [0]:
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(150, 150,3)),
    keras.layers.Dense(128, activation='relu'),
    
    keras.layers.Dense(2, activation='softmax'),
    
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(Images, Labels, epochs=30)
trained=model.fit(Images, Labels, epochs=30)
In [0]:
img=Images[2]
print(img.shape)
img = (np.expand_dims(img,0))
print(img.shape)
(150, 150, 3)
(1, 150, 150, 3)
In [0]:
plt.figure()
plt.imshow(Images[2])
plt.colorbar()
plt.grid(False)
plt.show()

In [0]:
predictions_single = model.predict(img)
print(predictions_single)
print(np.sum(predictions_single))
print(np.argmax(predictions_single))
print(class_names[np.argmax(predictions_single)])
[[0. 1.]]
1.0
1
Panda
In [0]:
img=cv2.imread("1014.jpg")
img_cvt=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_cvt)
plt.show()

In [0]:
img2=img_cvt
img2=cv2.resize(img2,(img_size,img_size))
print(img2.shape)
img2 = (np.expand_dims(img2,0))
print(img2.shape)
(150, 150, 3)
(1, 150, 150, 3)
In [0]:
predictions_single = model.predict(img2)
print(predictions_single)
print(np.sum(predictions_single))
print(np.argmax(predictions_single))
print(class_names[np.argmax(predictions_single)])
[[1.2754864e-31 1.0000000e+00]]
1.0
1
Panda
In [0]:
#Gracias por aprender más de TensorFlow
img=cv2.imread("abbi.jpg")
img_cvt=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_cvt)
plt.show()
