from django.urls import include, path
from .views import CustomLoginView, LoginAPI, RegisterAPI,mediapipePage,coursePage
from . import views
from django.contrib.auth.views import LogoutView
from rest_framework import routers
from knox import views as knox_views


urlpatterns = [
    path('', views.home, name='home'),
    path('api/register/', RegisterAPI.as_view(), name='register'),
    path('api/login/', LoginAPI.as_view(), name='login'),
    path('api/logout/', knox_views.LogoutView.as_view(), name='logout'),
    path('api/logoutall/', knox_views.LogoutAllView.as_view(), name='logoutall'),
    path('register/', views.registerPage, name="register"),
    path('login/', views.loginPage, name="login"),  
    path('mediapipePage/', mediapipePage, name='mediapipePage'),
    path('exercise/', coursePage, name='exercise'),
    path('feedback/', views.feedbackpage, name='feedback'),
    path('logout/', LogoutView.as_view(next_page='login'), name='logout'),
]