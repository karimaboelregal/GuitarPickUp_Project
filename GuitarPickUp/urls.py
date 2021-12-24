from django.urls import path
from .views import CustomLoginView,RegisterPage

urlpatterns = [
    path('', CustomLoginView.as_view(), name='login'),
    path('register/', RegisterPage.as_view(), name='register'),
]