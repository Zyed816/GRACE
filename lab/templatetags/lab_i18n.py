from django import template

from ..ui_text import build_language_switch_url, get_ui_language, text, ui_html_lang


register = template.Library()


@register.simple_tag(takes_context=True)
def tr(context, key, **kwargs):
    request = context.get("request")
    return text(key, get_ui_language(request), **kwargs)


@register.simple_tag(takes_context=True)
def current_lang(context):
    return get_ui_language(context.get("request"))


@register.simple_tag(takes_context=True)
def lang_url(context, language):
    return build_language_switch_url(context.get("request"), language)


@register.simple_tag(takes_context=True)
def html_lang_code(context):
    return ui_html_lang(get_ui_language(context.get("request")))
