from django.shortcuts import render

from .models import Photo


def get_photos(request):
    photos = Photo.objects.all()
    return render(request, 'app/archive.html', {'photos': photos})