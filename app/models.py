from django.db import models
from django.contrib.auth.models import User


# Добавим фото в пользователей
# class UserProfile(models.Model):
#     user = models.OneToOneField(User, on_delete=models.CASCADE)
#     telegram = models.CharField(max_length=150, blank=True, null=True)
#     avatar = models.ImageField(upload_to='images/avatars/', verbose_name='аватарки')
#
#     def __str__(self):
#         return str(self.user)
class UserProfile(models.Model):
  user = models.OneToOneField(User, on_delete=models.CASCADE)
  picture = models.TextField(null=True, blank=True)


class Tags(models.Model):
    name = models.CharField(max_length=20, db_index=True)
    slug = models.SlugField(max_length=20, unique=True)

    class Meta:
        verbose_name_plural = 'Tags'

    def __str__(self):
        return self.name


class Guidance(models.Model):
    guidance_id = models.IntegerField(primary_key=True, unique=True, db_index=True, auto_created=True)
    text = models.TextField(verbose_name='текст рекоммендации')
    slug = models.SlugField(max_length=50, unique=True)
    tag = models.ForeignKey(Tags, on_delete=models.CASCADE)
    prepopulated_fields = {'slug': ('text',)}  # формирование slug

    class Meta:
        verbose_name_plural = 'Guidance'

    def __str__(self):
        return self.slug


class Photo(models.Model):
    user = models.ForeignKey(User, default=True, on_delete=models.CASCADE)  # settings.AUTH_USER_MODEL
    guidance = models.ForeignKey(Guidance, null=True, on_delete=models.SET_NULL, blank=True)
    title = models.CharField(max_length=255, verbose_name='наименование')
    content = models.TextField(verbose_name='описание')
    photo_file = models.ImageField(upload_to='images/', verbose_name='изображение')
    slug = models.SlugField(max_length=50, verbose_name='краткое наименование')
    diaphragm = models.CharField(max_length=10, default="", blank=True, verbose_name='диафрагма')
    exposure = models.CharField(max_length=10, default="", blank=True, verbose_name='выдержка')
    iso = models.CharField(max_length=10, default="", blank=True, verbose_name='ISO')
    raiting_light = models.DecimalField(decimal_places=0, max_digits=100, null=True, blank=True,
                                        verbose_name='рейтинг свет')
    raiting_focus = models.DecimalField(decimal_places=0, max_digits=100, null=True, blank=True,
                                        verbose_name='рейтинг фокус')
    raiting_composition = models.DecimalField(decimal_places=0, max_digits=100, null=True, blank=True,
                                              verbose_name='рейтинг композиция')
    raiting_result = models.DecimalField(decimal_places=0, max_digits=100, null=True, blank=True,
                                         verbose_name='итоговый рейтинг')
    published_date = models.DateTimeField(auto_now=False, auto_now_add=True)

    prepopulated_fields = {'slug': ('title',), }  # формирование slug

    class Meta:
        verbose_name_plural = 'Photos'
        ordering = ['-published_date']

    def __unicode__(self):
        return self.title

    def get_abs_url(self):
        return "/all/"
