from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("experiments", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="PipelineResult",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("dataset", models.CharField(max_length=32)),
                ("method_key", models.CharField(max_length=32)),
                ("method_name", models.CharField(blank=True, max_length=64)),
                ("stage", models.CharField(choices=[("baseline", "Baseline"), ("top_verify", "Top Verify"), ("summary", "Summary")], default="summary", max_length=32)),
                ("candidate_rank", models.PositiveIntegerField(blank=True, null=True)),
                ("run_idx", models.PositiveIntegerField(blank=True, null=True)),
                ("F1Mi_mean", models.FloatField(blank=True, null=True)),
                ("F1Mi_std", models.FloatField(blank=True, null=True)),
                ("F1Ma_mean", models.FloatField(blank=True, null=True)),
                ("F1Ma_std", models.FloatField(blank=True, null=True)),
                ("robust_score", models.FloatField(blank=True, null=True)),
                ("delta_vs_grace", models.FloatField(blank=True, null=True)),
                ("params_json", models.JSONField(blank=True, default=dict)),
                ("notes", models.TextField(blank=True)),
                ("source_csv", models.CharField(blank=True, max_length=512)),
                ("created_time", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "ordering": ["dataset", "method_key", "stage", "candidate_rank", "run_idx"],
            },
        ),
        migrations.AddIndex(
            model_name="pipelineresult",
            index=models.Index(fields=["dataset", "method_key", "stage"], name="experiments_dataset_6f4ad0_idx"),
        ),
    ]