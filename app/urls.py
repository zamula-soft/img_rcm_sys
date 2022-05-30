from django.urls import path
from . import views

app_name = 'app'

urlpatterns = [
    path('old', views.get_photos_old, name='get_photos_old'),
    path('', views.get_main_page, name='get_main_page'),
    path('photo/<slug:slug>/', views.get_photo_detail, name='get_photo_detail'),
    path('personal/', views.get_personal_cabinet, name='get_personal_cabinet'),
    path('archive/', views.get_my_photos, name='get_my_photos'),
    path('new_photo/', views.get_new_photo, name='get_new_photo'),
    # path('accounts/profile/', views.UpdateCoverImageForm, name = 'profile'),
]
