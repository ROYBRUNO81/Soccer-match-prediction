# predictor/urls.py
from django.urls import path
from . import views

app_name = 'predictor'

urlpatterns = [
    path('',         views.index,   name='index'),
    path('predict/', views.predict, name='predict'),
]