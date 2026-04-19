from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="ExperimentRun",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(blank=True, max_length=120, verbose_name="Run Name")),
                (
                    "experiment_type",
                    models.CharField(
                        choices=[
                            ("method_comparison", "Method Comparison"),
                            ("sampling_bias", "Sampling Bias"),
                            ("sensitivity", "Sensitivity Analysis"),
                        ],
                        max_length=32,
                        verbose_name="Experiment Type",
                    ),
                ),
                ("dataset", models.CharField(blank=True, max_length=32, verbose_name="Dataset")),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("pending", "Pending"),
                            ("running", "Running"),
                            ("succeeded", "Succeeded"),
                            ("failed", "Failed"),
                        ],
                        default="pending",
                        max_length=16,
                        verbose_name="Status",
                    ),
                ),
                ("config", models.JSONField(blank=True, default=dict, verbose_name="Config")),
                ("command", models.TextField(blank=True, verbose_name="Command")),
                ("stdout_log", models.TextField(blank=True, verbose_name="Stdout Log")),
                ("error_message", models.TextField(blank=True, verbose_name="Error Message")),
                ("result_summary", models.JSONField(blank=True, default=dict, verbose_name="Result Summary")),
                ("created_at", models.DateTimeField(auto_now_add=True, verbose_name="Created At")),
                ("started_at", models.DateTimeField(blank=True, null=True, verbose_name="Started At")),
                ("finished_at", models.DateTimeField(blank=True, null=True, verbose_name="Finished At")),
            ],
            options={
                "verbose_name": "Experiment Run",
                "verbose_name_plural": "Experiment Runs",
                "ordering": ["-created_at"],
            },
        ),
        migrations.CreateModel(
            name="ExperimentArtifact",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("label", models.CharField(max_length=120, verbose_name="Label")),
                (
                    "artifact_type",
                    models.CharField(
                        choices=[("csv", "CSV"), ("image", "Image"), ("report", "Report"), ("other", "Other")],
                        default="other",
                        max_length=16,
                        verbose_name="Artifact Type",
                    ),
                ),
                ("relative_path", models.CharField(max_length=255, verbose_name="Relative Path")),
                ("metadata", models.JSONField(blank=True, default=dict, verbose_name="Metadata")),
                ("created_at", models.DateTimeField(auto_now_add=True, verbose_name="Created At")),
                (
                    "run",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="artifacts",
                        to="lab.experimentrun",
                    ),
                ),
            ],
            options={
                "verbose_name": "Experiment Artifact",
                "verbose_name_plural": "Experiment Artifacts",
                "ordering": ["artifact_type", "id"],
            },
        ),
    ]
