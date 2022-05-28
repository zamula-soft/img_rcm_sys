from django.contrib import admin

from .models import Tags, Guidance, Photo, UserProfile


@admin.register(Tags)
@admin.register(Guidance)
class GuidanceAdmin(admin.ModelAdmin):
    list_display = ['slug']


@admin.register(Photo)
class PhotoAdmin(admin.ModelAdmin):
    search_fields = ['title', 'slug']
    list_display = ['title', 'published_date', 'raiting_light', 'raiting_composition', 'raiting_focus',
                    'raiting_result']


@admin.register(UserProfile)
class UserProfile(admin.ModelAdmin):
    list_display = ['picture']