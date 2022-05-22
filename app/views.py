from django.shortcuts import render, get_object_or_404

from .models import Photo


def get_photos_old(request):
    photos = Photo.objects.all()
    return render(request, 'app/archive.html', {'photos': photos})


def get_my_photos(request):
    photos = Photo.objects.all()
    return render(request, 'app/my_photos.html', {'photos': photos})


def photo_detail(request, slug):
    photo = get_object_or_404(Photo, slug=slug)
    return render(request, 'app/photos/detail.html', {'photo': photo})