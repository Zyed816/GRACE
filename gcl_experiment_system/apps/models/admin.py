from django.contrib import admin

from .models import MethodProfile


@admin.register(MethodProfile)
class MethodProfileAdmin(admin.ModelAdmin):
    list_display = ("display_name", "key")
