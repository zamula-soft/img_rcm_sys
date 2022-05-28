# USERS

def update_user_social_data(strategy, *args, **kwargs):
  response = kwargs['response']
  backend = kwargs['backend']
  user = kwargs['user']

  if response['picture']:
    url = response['picture']
    print(url)
    userProfile_obj = UserProfile()
    userProfile_obj.user = user
    userProfile_obj.picture = url
    userProfile_obj.save()
    print("ОК")


def get_avatar(backend, response, user=None, *args, **kwargs):
    url = None
    print('begin')
    if backend.name == 'vk-oauth2' or backend.name == "google-oauth2":
        url = response.get('photo', '')

    if url:
        # user.avatar = url
        # user.save()
        print(user.avatar)