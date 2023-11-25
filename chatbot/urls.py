from django.urls import path,include
from .import views

urlpatterns = [
    path('', views.Homepage, name='Homepage'),
    path('review/', views.review_entry, name='review_entry'),
    path('get_doctors/',views.get_doctors_by_department, name='get_doctors_by_department'),
    path('ask_me/',views.ask_me,name='ask_me'),
    path('get_doctor_details/<int:id>/',views.get_doctor_details,name='get_doctor_details'),
    path('user-question/',views.user_question,name='get_doctor_details')
]