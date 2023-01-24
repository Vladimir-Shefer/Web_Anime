from django.db import models

class ImageInfo(models.Model):
   isAnime  = models.BooleanField()
   picture = models.ImageField(upload_to = 'pictures')

   class Meta:
      db_table = "images"

