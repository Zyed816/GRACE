from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="MethodProfile",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("key", models.SlugField(max_length=32, unique=True)),
                ("display_name", models.CharField(max_length=64)),
                ("description", models.TextField()),
                ("architecture", models.TextField()),
                ("key_parameters", models.JSONField(blank=True, default=list)),
            ],
            options={"ordering": ["display_name"]},
        ),
    ]
