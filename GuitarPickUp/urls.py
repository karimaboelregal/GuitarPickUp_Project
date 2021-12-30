from django.urls import path
from .views import CustomLoginView,RegisterPage,mediapipePage
from . import views
from django.contrib.auth.views import LogoutView

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', CustomLoginView.as_view(), name='login'),
    path('register/', RegisterPage.as_view(), name='register'),
    path('show/', mediapipePage, name='show'),
    path('logout/', LogoutView.as_view(next_page='login'), name='logout'),
]