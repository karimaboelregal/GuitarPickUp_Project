from django.urls import path
from .views import CustomLoginView,mediapipePage,coursePage
from . import views
from django.contrib.auth.views import LogoutView

 
urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.registerPage, name="register"),
    path('login/', views.loginPage, name="login"),  
    path('mediapipePage/', mediapipePage, name='mediapipePage'),
    path('exercise/', coursePage, name='exercise'),
    path('feedback/', views.feedbackpage, name='feedback'),
    path('logout/', LogoutView.as_view(next_page='login'), name='logout'),
]