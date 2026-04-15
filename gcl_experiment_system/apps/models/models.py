from django.db import models


class MethodProfile(models.Model):
    key = models.SlugField(max_length=32, unique=True)
    display_name = models.CharField(max_length=64)
    description = models.TextField()
    architecture = models.TextField()
    key_parameters = models.JSONField(default=list, blank=True)

    class Meta:
        ordering = ["display_name"]

    def __str__(self):
        return self.display_name
