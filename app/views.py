from django.shortcuts import render

from .models import Photo


def get_photos_old(request):
    photos = Photo.objects.all()
    return render(request, 'app/archive.html', {'photos': photos})


def get_my_photos(request):
    photos = Photo.objects.all()
    return render(request, 'app/my_photos.html', {'photos': photos})