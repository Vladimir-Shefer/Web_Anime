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


print(len(tf.config.list_logical_devices('GPU')))