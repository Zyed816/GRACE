from django.http import Http404
from django.shortcuts import render

from .catalog import METHOD_CATALOG


def method_list(request):
    methods = list(METHOD_CATALOG.values())
    return render(request, "models/list.html", {"methods": methods})


def method_detail(request, key):
    method = METHOD_CATALOG.get(key)
    if method is None:
        raise Http404(f"Unknown method: {key}")
    return render(request, "models/detail.html", {"method": method})
