from django.urls import path

from . import views

app_name = 'objects'

urlpatterns = [
    path('', views.index, name='index'),
]
