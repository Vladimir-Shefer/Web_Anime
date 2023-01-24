from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('polls/', include('polls.urls')),
    path('i/', include('image_load.urls')),
    path('admin/', admin.site.urls),
]