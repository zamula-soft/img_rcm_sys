from django.urls import path
from . import views

app_name = 'app'

urlpatterns = [
    path('', views.get_photos, name='get_photos'),
    path('my', views.get_my_photos, name='get_my_photos'),
]

