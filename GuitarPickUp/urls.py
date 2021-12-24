from django.urls import path
from .views import CustomLoginView,RegisterPage
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', CustomLoginView.as_view(), name='login'),
    path('register/', RegisterPage.as_view(), name='register'),
]