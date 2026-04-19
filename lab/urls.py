from django.urls import path

from . import views


app_name = "lab"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("experiments/method-comparison/create/", views.create_method_comparison_run, name="create_method_comparison"),
    path("experiments/sampling-bias/create/", views.create_sampling_bias_run, name="create_sampling_bias"),
    path("experiments/sensitivity/create/", views.create_sensitivity_run, name="create_sensitivity"),
    path("runs/<int:pk>/", views.run_detail, name="run_detail"),
    path("runs/<int:pk>/stop/", views.stop_run, name="stop_run"),
    path("runs/<int:pk>/delete/", views.delete_run, name="delete_run"),
    path("runs/<int:run_id>/artifacts/<int:artifact_id>/", views.artifact_file, name="artifact_file"),
]
