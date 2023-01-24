#-*- coding: utf-8 -*-
from operator import truediv
from django.conf import settings
from django.core.files.base import ContentFile
import PIL
import PIL.ImageOps   
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist         # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
import django
from django.forms import ImageField, NullBooleanField
from django.http import HttpResponse
from django.shortcuts import render
from image_load.forms import ImageForm
from image_load.models import ImageInfo
from keras.preprocessing import image
import tensorflow as tf
from django.http import JsonResponse
from io import BytesIO
from PIL import Image , ImageFilter
import re
import base64
import json
from django.apps import apps as django_apps
from serpapi import GoogleSearch
import requests

loaded_model = load_model("anime_not_anime.h5")
loaded_model_gen = load_model("anime_gen.h5")
def SaveImage(request):
  
  
   if request.is_ajax():
    if request.method == 'GET':
        noice = np.random.randn(32,100)
        pred = loaded_model_gen.predict(noice)
        formatted = (pred[1] * 255 / np.max(pred[1])).astype('uint8')
        img = Image.fromarray(formatted)
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        img = img.filter(ImageFilter.SHARPEN)


        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue())
        img_str = str(img_str)
        img_str = img_str[2:]
        l = len(img_str)

        img_str = img_str[:l-1] 
        print(img_str)
        return JsonResponse({'successff':img_str, 'errorMsg':""})
    elif request.method == 'POST':
        '''
        params = {
            "q": "human portrait",
            "tbm": "isch",
            "ijn": "0",
            "api_key": "lol_it_is_private_key"
                }

        search = GoogleSearch(params)
        results = search.get_dict()
        images_results = results["images_results"]
        
        response_image = requests.get(images_results[1]["thumbnail"])
        img34 = Image.open(BytesIO(response_image.content))

        '''
        img1 = request.POST.get("image", None)
       
        image_data = re.sub("^data:image/png;base64,", "", img1)
        image_data = base64.b64decode(image_data)
      
        image_data = BytesIO(image_data)
        img = Image.open(image_data)
      
        thumb_io = BytesIO()
        thumb_io1 = BytesIO()
        img.save(thumb_io, img.format, quality=60)
        ''' 
        img34.save(thumb_io1, img34.format, quality=60)
        '''

        i = ImageInfo()
        
        i.isAnime = True
        i.picture.save("pictures/"+img.filename, ContentFile(thumb_io.getvalue()), save=False)
        i.save()
        '''
     
        i.picture.save("pictures/"+img34.filename, ContentFile(thumb_io1.getvalue()), save=False)
        i.save()

        '''

        #img.save('pictures/i.png')
      



      
      #img2 = tf.io.read_file('pictures/i.png')
        path = i.picture.path
        img2 = tf.io.read_file(path)
      
      
    
        tensor = tf.io.decode_image(img2, channels=3, dtype=tf.dtypes.float32)
      
               
        tensor = tf.image.resize(tensor, [64, 64])


      
        input_tensor = tf.expand_dims(tensor, axis=0)

     


        a = np.argmax(loaded_model.predict(input_tensor))
        print(loaded_model.predict(input_tensor))
        if(a==1):
            i.isAnime = False
        else:
            i.isAnime = True
        print(a)
        i.save()
        print(i.isAnime)
        return JsonResponse({'successff':str(a), 'errorMsg':""})
      
               

   else:      
      correct = True
      
      return render(request, 'image.html', locals())
