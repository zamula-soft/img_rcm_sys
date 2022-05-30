import messages as messages
from django.contrib.auth.models import User
from django.http import Http404
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages

# from .forms import UpdateCoverImageForm
from .models import Photo
    # , UserProfile


def get_photos_old(request):
    photos = Photo.objects.all()
    return render(request, 'app/archive.html', {'photos': photos})

def get_main_page(request):
    context = {
        # 'photo': get_object_or_404(Photo, slug=slug)
        # if request.user.is_authenticated else []
    }
    return render(request, 'app/index.html', context)


def get_my_photos(request):
    photos = Photo.objects.all()
    return render(request, 'app/photos/my_photos.html', {'photos': photos})


def get_photo_detail(request, slug):
    context = {
        'photo': get_object_or_404(Photo, slug=slug)
        if request.user.is_authenticated else []
    }

    return render(request, 'app/photos/detail.html', context)


# personal cabinet
def get_personal_cabinet(request):
    context = {
        'photos': Photo.objects.all()
        if request.user.is_authenticated else []
    }
    return render(request, 'app/personal_cabinet.html', context)


def get_new_photo(request):
    context = {
        # 'photo': get_object_or_404(Photo, slug=slug)
        # if request.user.is_authenticated else []
    }
    return render(request, 'app/photos/new_photo.html', context)


# def profile(request, username):
#     try:
#         user = User.objects.get(username=username)
#         user_ = User.objects.filter(username=username)
#         memer = UserProfile.objects.filter(user=user_[0].id)
#     except User.DoesNotExist:
#         raise Http404("Memer does not exist.")
#
#     context = {
#         'user_': user_,
#         'memer': memer,
#     }
#     if request.method == "POST":
#         # bioForm = EditBioForm(data=request.POST, instance=request.user.memer)
#         coverImageForm = UpdateCoverImageForm(data=request.FILES, instance=request.user.memer)
#         if coverImageForm.is_valid():
#             memer_ = coverImageForm.save(commit=False)
#             memer_.save()
#             messages.success(request, "Cover Image has been updated successfully!")
#             # print(coverImageForm)
#             return redirect('/profile/'+user_[0].username)
#         else:
#             messages.error(request, "Something wrong happend")
#             return redirect('/profile/'+user_[0].username)
#     return render(request, 'profile.html', context)