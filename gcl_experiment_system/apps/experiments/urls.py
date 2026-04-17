from django.urls import path

from . import views


urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("artifacts/<str:bucket>/<path:filename>", views.artifact_file, name="artifact-file"),
    path("experiments/new/", views.experiment_create, name="experiment-create"),
    path("experiments/<int:pk>/", views.experiment_detail, name="experiment-detail"),
    path("experiments/<int:pk>/start/", views.experiment_start, name="experiment-start"),
    path("experiments/<int:pk>/stop/", views.experiment_stop, name="experiment-stop"),
    path("experiments/<int:pk>/monitor/", views.experiment_monitor, name="experiment-monitor"),
    path("history/", views.experiment_history, name="experiment-history"),
    path("results/", views.results_overview, name="results-overview"),
    path("api/experiments/create/", views.api_experiment_create, name="api-experiment-create"),
    path("api/experiments/<int:pk>/", views.api_monitor, name="api-monitor"),
    path("api/experiments/<int:pk>/stop/", views.api_experiment_stop, name="api-experiment-stop"),
]
