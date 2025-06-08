# detector_app/urls.py
from django.urls import path
from . import views

app_name = 'detector_app'

urlpatterns = [
    path('', views.index_view, name='index'),
    path('predict/', views.predict_video_view, name='predict_video'),

    # Authentication URLs
    path('login/', views.login_view, name='login_view'),
    path('login/action/', views.login_action_view, name='login_action'), # Handles login form submission
    path('signup/', views.signup_view, name='signup_view'),
    path('signup/action/', views.signup_action_view, name='signup_action'), # Handles signup form submission
    path('logout/', views.logout_view, name='logout_view'),
]