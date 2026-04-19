from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("lab", "0001_initial"),
    ]

    operations = [
        migrations.AlterField(
            model_name="experimentrun",
            name="status",
            field=models.CharField(
                choices=[
                    ("pending", "Pending"),
                    ("running", "Running"),
                    ("succeeded", "Succeeded"),
                    ("failed", "Failed"),
                    ("aborted", "Aborted"),
                ],
                default="pending",
                max_length=16,
                verbose_name="Status",
            ),
        ),
        migrations.AddField(
            model_name="experimentrun",
            name="worker_pid",
            field=models.IntegerField(blank=True, null=True, verbose_name="Worker PID"),
        ),
    ]
