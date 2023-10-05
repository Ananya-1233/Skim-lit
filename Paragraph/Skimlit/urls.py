from django.urls import path
from Skimlit.views import predict

urlpatterns = [
    path('', predict, name='predict'),
]