from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Experiment",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("dataset", models.CharField(max_length=32)),
                ("model_name", models.CharField(max_length=32)),
                ("learning_rate", models.FloatField(default=0.01)),
                ("hidden_dim", models.PositiveIntegerField(default=256)),
                ("epochs", models.PositiveIntegerField(default=200)),
                ("temperature", models.FloatField(default=0.5)),
                ("drop_edge_rate", models.FloatField(default=0.2)),
                ("drop_feature_rate", models.FloatField(default=0.2)),
                ("extra_params", models.JSONField(blank=True, default=dict)),
                ("final_accuracy", models.FloatField(blank=True, null=True)),
                ("final_f1mi", models.FloatField(blank=True, null=True)),
                ("final_f1ma", models.FloatField(blank=True, null=True)),
                ("run_seconds", models.FloatField(blank=True, null=True)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("pending", "Pending"),
                            ("running", "Running"),
                            ("succeeded", "Succeeded"),
                            ("failed", "Failed"),
                            ("cancelled", "Cancelled"),
                        ],
                        default="pending",
                        max_length=16,
                    ),
                ),
                ("stdout_path", models.CharField(blank=True, max_length=512)),
                ("exp1_log_path", models.CharField(blank=True, max_length=512)),
                ("created_time", models.DateTimeField(auto_now_add=True)),
                ("started_time", models.DateTimeField(blank=True, null=True)),
                ("finished_time", models.DateTimeField(blank=True, null=True)),
                ("error_message", models.TextField(blank=True)),
            ],
            options={"ordering": ["-created_time"]},
        ),
        migrations.CreateModel(
            name="ExperimentMetric",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(max_length=64)),
                ("value", models.FloatField()),
                ("step", models.PositiveIntegerField(default=0)),
                (
                    "experiment",
                    models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="metrics", to="experiments.experiment"),
                ),
            ],
            options={"ordering": ["name", "step"]},
        ),
        migrations.CreateModel(
            name="ExperimentLog",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("epoch", models.PositiveIntegerField()),
                ("loss", models.FloatField()),
                ("accuracy", models.FloatField(blank=True, null=True)),
                ("payload", models.JSONField(blank=True, default=dict)),
                ("created_time", models.DateTimeField(auto_now_add=True)),
                (
                    "experiment",
                    models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="logs", to="experiments.experiment"),
                ),
            ],
            options={"ordering": ["epoch"]},
        ),
    ]
