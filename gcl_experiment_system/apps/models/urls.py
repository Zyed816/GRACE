from django.urls import path

from . import views


urlpatterns = [
    path("", views.method_list, name="method-list"),
    path("<str:key>/", views.method_detail, name="method-detail"),
]
