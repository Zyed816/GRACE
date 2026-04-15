from django.contrib import admin
from django.urls import include, path


urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("apps.experiments.urls")),
    path("datasets/", include("apps.datasets.urls")),
    path("models/", include("apps.models.urls")),
]
