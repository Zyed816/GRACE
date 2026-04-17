from django.db import migrations, models


def backfill_task_type(apps, schema_editor):
    Experiment = apps.get_model("experiments", "Experiment")
    Experiment.objects.filter(task_type="").update(task_type="train")


class Migration(migrations.Migration):

    dependencies = [
        ("experiments", "0002_pipelineresult"),
    ]

    operations = [
        migrations.AddField(
            model_name="experiment",
            name="artifacts",
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name="experiment",
            name="cancel_requested",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="experiment",
            name="extra_cli_args",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="experiment",
            name="task_params",
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name="experiment",
            name="task_type",
            field=models.CharField(
                choices=[
                    ("train", "Train"),
                    ("grid_search", "Grid Search"),
                    ("top_verify", "Top Verify"),
                    ("full_pipeline_single", "Full Pipeline (Single Dataset)"),
                    ("full_pipeline_multi", "Full Pipeline (Multi Dataset)"),
                ],
                default="train",
                max_length=32,
            ),
        ),
        migrations.RunPython(backfill_task_type, migrations.RunPython.noop),
    ]
