from .views import *
from django.urls import path, include

urlpatterns = [
    path('file/', firstFileView),
    path('file/<str:col>/', diffStyleInferenceView),
    path('file/re/<str:image_col>/', sameStyleInferenceView),
    path('image/<str:image_col>/', fileGetView),
    path('history/<str:user_code>/', historyView),
    path('list/', listView),
]
