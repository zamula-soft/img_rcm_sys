from django.urls import path
from . import views

app_name = 'app'

urlpatterns = [
    path('old', views.get_photos_old, name='get_photos_old'),
    path('', views.get_my_photos, name='get_my_photos'),
    path('photo/<slug:slug>/', views.photo_detail, name='photo_detail')
]
