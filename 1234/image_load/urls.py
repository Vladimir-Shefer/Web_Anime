
from django.conf import settings
from django.urls import path
from image_load import views
from . import views
from django.conf.urls import url
from django.views.generic import TemplateView
from django.template.context import RequestContext

from django.conf.urls.static import static
urlpatterns = [
     path('', views.SaveImage, name='imaged'),
     
     
    
 ]

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)