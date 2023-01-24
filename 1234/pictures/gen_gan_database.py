import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.utils.np_utils import to_categorical
from os import listdir
import tensorflow.keras.layers as L
import numpy as np
from PIL import Image 
from keras.preprocessing.image import ImageDataGenerator
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    '/home/vladimir/Рабочий стол/images/', label_mode=None, image_size=(64,64),batch_size=128
)
dataset = dataset.map(lambda x: (x / 127.5) - 1)
loss = keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real,fake):
  real_loss = loss(tf.ones_like(real),real)
  fake_loss = loss(tf.zeros_like(fake),fake)
  return real_loss + fake_loss
def generator_loss(fake):
  gen_loss = loss(tf.ones_like(fake),fake)
  return gen_loss
generator_optimizer = keras.optimizers.Adam(0.0002,0.5)
discriminator_optimizer = keras.optimizers.Adam(0.0002,0.5)

@tf.function
def train_step(images):
    noise = tf.random.normal([256,100])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)
      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)
      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss,disc_loss

def train(dataset, epochs):
  gen_loss=0
  dis_loss=0
  gen_plot = []
  dis_plot = []
  for epoch in range(epochs):
    print(f'{epoch+1} of {epochs}',end = ' ')
    print(f'generator loss is {gen_loss} and discriminator loss is {dis_loss}')
    for image_batch in dataset: 
      gen_loss,dis_loss = train_step(image_batch)
    gen_plot.append(gen_loss)
    dis_plot.append(dis_loss)
  x = [X for X in range(epochs)]
  plt.plot(x,gen_plot)
  plt.plot(x,dis_plot)
  plt.title('epoch vs loss')
  plt.legend(['gen_loss','dis_loss'])
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.show()

code_size = 100
batch_size = 128
generator = keras.Sequential()
generator.add(L.Input(shape=(100,)))
generator.add(L.Dense(4*4*256))
generator.add(L.BatchNormalization())
generator.add(L.ReLU())
generator.add(L.Reshape((4,4,256)))
generator.add(L.Conv2DTranspose(128,kernel_size=4,padding='same',strides=2))
generator.add(L.BatchNormalization())
generator.add(L.ReLU())
generator.add(L.Conv2DTranspose(64,kernel_size=4,padding='same',strides=2))
generator.add(L.BatchNormalization())
generator.add(L.ReLU())
generator.add(L.Conv2DTranspose(32,kernel_size=4,padding='same',strides=2))
generator.add(L.BatchNormalization())
generator.add(L.ReLU())
generator.add(L.Conv2DTranspose(3,kernel_size=4,padding='same',strides=2))
generator.add(L.Activation('tanh'))


image_shape = (64,64,3)
discriminator = keras.Sequential()
discriminator.add(L.Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
discriminator.add(L.LeakyReLU(alpha=0.2))
discriminator.add(L.Dropout(0.25))
discriminator.add(L.Conv2D(64, kernel_size=3, strides=2, padding="same"))
discriminator.add(L.ZeroPadding2D(padding=((0,1),(0,1))))
discriminator.add(L.BatchNormalization(momentum=0.8))
discriminator.add(L.LeakyReLU(alpha=0.2))
discriminator.add(L.Dropout(0.25))
discriminator.add(L.Conv2D(128, kernel_size=3, strides=2, padding="same"))
discriminator.add(L.BatchNormalization(momentum=0.8))
discriminator.add(L.LeakyReLU(alpha=0.2))
discriminator.add(L.Dropout(0.25))
discriminator.add(L.Conv2D(256, kernel_size=3, strides=1, padding="same"))
discriminator.add(L.BatchNormalization(momentum=0.8))
discriminator.add(L.LeakyReLU(alpha=0.2))
discriminator.add(L.Dropout(0.25))
discriminator.add(L.Flatten())
discriminator.add(L.Dense(1))

train(dataset,20)
noice = np.random.randn(32,100)
pred = generator.predict(noice)
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
w=10
h=10
fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 5
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(pred[i])
plt.show()

generator.save("anime_gen.h5")
