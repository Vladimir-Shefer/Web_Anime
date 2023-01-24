import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.utils.np_utils import to_categorical
from os import listdir
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
import numpy as np
from PIL import Image 
from keras.preprocessing.image import ImageDataGenerator
val_ds = tf.keras.utils.image_dataset_from_directory(
  "/home/vladimir/Рабочий стол/data2/",
  validation_split=0.2,
  subset="training",
  labels = 'inferred',
  label_mode =  'categorical',
  seed=123,
  image_size=(64, 64),
  batch_size=32) 
for el in val_ds:
    print(el[1])
'''

images_ar = ()
labels = ()
i = 1
# get the path/directory
folder_dir = "/media/vladimir/Elements/for project/im/images"
for images in os.listdir(folder_dir):
 
    # check if the image ends with png
    if (images.endswith(".jpg")):
        input_path = os.path.join(folder_dir, images)
        img2 = tf.io.read_file(input_path)
        print(images)
        tensor = tf.io.decode_image(img2, channels=3, dtype=tf.dtypes.float32)
      
               
        tensor = tf.image.resize(tensor, [64, 64])
        

      
        input_tensor = tf.expand_dims(tensor, axis=0)
        #input_tensor = input_tensor/255
        np.append(images_ar, input_tensor, 0)
        #images_ar.append(input_tensor)
        a = np.array([1])
        tensor1 = tf.constant(a)
        np.append(labels, a, 0)
        #labels.append(tensor1)
        i = i+1
        if (i>100):
            break
# get the path/directory
folder_dir = "/media/vladimir/Elements/for project/im/people"
for images in os.listdir(folder_dir):
 
    # check if the image ends with png
    if (images.endswith(".jpg")):
        input_path = os.path.join(folder_dir, images)
        img2 = tf.io.read_file(input_path)
        print(images)
        tensor = tf.io.decode_image(img2, channels=3, dtype=tf.dtypes.float32)
      
               
        tensor = tf.image.resize(tensor, [64, 64])300
        

      
        input_tensor = tf.expand_dims(tensor, axis=0)
        #input_tensor = input_tensor/255
        np.append(images_ar, input_tensor, 0)
        #images_ar.append(input_tensor)
        a = np.array([0])
        
        tensor1 = tf.constant(a)
        np.append(labels, a, 0)
        #labels.append(tensor1)
        i = i+1
        if (i>200):
            break

print(type(images_ar))
dataset = tf.data.Dataset.from_tensor_slices((images_ar, labels))
dataset = dataset.shuffle(i)
print(dataset)

# a = np.repeat(1,63565)
# print(a)
# # Y_one_hot=to_categorical(a, num_classes=2)
# # print (Y_one_hot)
# dataset = keras.preprocessing.image_dataset_from_directory(
#     "/media/vladimir/Elements/for project/im/w",labels=  np.array([[1],[1],[1]]), label_mode = 'categorical', image_size=(64, 64), batch_size=128
# )
# dataset = dataset.map(lambda x: x / 255.0)

# files = os.listdir("/media/vladimir/Elements/for project/im/images")   


discriminator = keras.Sequential(
    [
        keras.Input(shape=(64, 64, 3)),
        layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        # layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        # layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dropout(0.2),  
   
        layers.Dense(1,  activation='softmax')
    ],
    name="discriminator",
)
discriminator.summary()

discriminator.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

# a = tf.data.Dataset(
    
# )
# for images in dataset.take(1):  # only take first element of dataset
#     numpy_images = images.numpy()
#     df=pd.DataFrame({'image': image_file_paths, 'label':label_file_paths}).astype(str)

# a = [np.empty(np.shape(1)) for _ in range(63565)]

# Y_one_hot=to_categorical(a, num_classes=2)
# dataset2 = tf.data.Dataset.from_tensor_slices(Y_one_hot)
#dataset =  tf.data.Dataset.zip((dataset, dataset2))
#his = discriminator.fit(dataset, batch_size=32, epochs=5)
his = discriminator.fit(images_ar, labels, batch_size = 32, epochs = 2)
discriminator.save("anime_not_anime.h5")
'''

discriminator = keras.Sequential([
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(64, (3,3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(2,  activation='softmax')
])
discriminator.summary()

discriminator.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

history=discriminator.fit(val_ds,  epochs=5, verbose=1,
                 shuffle=True,  initial_epoch=0) 

discriminator.evaluate(val_ds)
discriminator.save("anime_human5.h5")