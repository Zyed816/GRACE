from html import escape

from django.utils.safestring import mark_safe


def _dynamic_range(values):
    if not values:
        return 0.0, 1.0
    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum
    padding = max(span * 0.2, 0.02)
    lower = minimum - padding
    upper = maximum + padding
    if lower == upper:
        upper = lower + 1.0
    return lower, upper


def render_bar_chart(title, points, width=760, height=320):
    if not points:
        return ""

    plot_x = 68
    plot_y = 32
    plot_width = width - 102
    plot_height = height - 82
    values = [float(point["value"]) for point in points]
    lower, upper = _dynamic_range(values)
    bar_width = plot_width / max(len(points) * 1.6, 1)
    gap = bar_width * 0.6
    axis_y = plot_y + plot_height
    colors = ["#0f766e", "#1d4ed8", "#b45309", "#7c3aed", "#be123c"]

    bars = []
    labels = []
    numbers = []
    for index, point in enumerate(points):
        value = float(point["value"])
        label = str(point["label"])
        bar_height = ((value - lower) / (upper - lower)) * plot_height
        x = plot_x + gap + index * (bar_width + gap)
        y = axis_y - bar_height
        color = colors[index % len(colors)]
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" '
            f'rx="12" fill="{color}" opacity="0.88" />'
        )
        labels.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{axis_y + 24:.2f}" '
            f'text-anchor="middle" font-size="11" fill="#334155">{escape(label)}</text>'
        )
        numbers.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{y - 8:.2f}" '
            f'text-anchor="middle" font-size="11" fill="#0f172a">{value:.4f}</text>'
        )

    ticks = []
    for tick_idx in range(5):
        y_value = lower + (upper - lower) * (4 - tick_idx) / 4
        y = plot_y + plot_height * tick_idx / 4
        ticks.append(
            f'<line x1="{plot_x}" y1="{y:.2f}" x2="{plot_x + plot_width}" y2="{y:.2f}" '
            f'stroke="#cbd5e1" stroke-dasharray="4 6" />'
        )
        ticks.append(
            f'<text x="{plot_x - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="10.5" fill="#64748b">'
            f"{y_value:.4f}</text>"
        )

    svg = f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">
      <rect x="0" y="0" width="{width}" height="{height}" rx="28" fill="#f8fafc" />
      <text x="{plot_x}" y="20" font-size="15" font-weight="700" fill="#0f172a">{escape(title)}</text>
      <line x1="{plot_x}" y1="{axis_y}" x2="{plot_x + plot_width}" y2="{axis_y}" stroke="#94a3b8" />
      <line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{axis_y}" stroke="#94a3b8" />
      {''.join(ticks)}
      {''.join(bars)}
      {''.join(numbers)}
      {''.join(labels)}
    </svg>
    """
    return mark_safe(svg)


def render_line_chart(title, points, x_key, y_key, stroke, width=760, height=320):
    if not points:
        return ""

    plot_x = 60
    plot_y = 28
    plot_width = width - 86
    plot_height = height - 70
    values = [float(point[y_key]) for point in points]
    lower, upper = _dynamic_range(values)
    axis_y = plot_y + plot_height

    polyline_points = []
    markers = []
    labels = []
    for index, point in enumerate(points):
        x_ratio = index / max(len(points) - 1, 1)
        x = plot_x + plot_width * x_ratio
        value = float(point[y_key])
        y = axis_y - ((value - lower) / (upper - lower)) * plot_height
        polyline_points.append(f"{x:.2f},{y:.2f}")
        markers.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.6" fill="{stroke}" />')

    for tick_idx in range(5):
        y_value = lower + (upper - lower) * (4 - tick_idx) / 4
        y = plot_y + plot_height * tick_idx / 4
        labels.append(
            f'<line x1="{plot_x}" y1="{y:.2f}" x2="{plot_x + plot_width}" y2="{y:.2f}" '
            f'stroke="#cbd5e1" stroke-dasharray="4 6" />'
        )
        labels.append(
            f'<text x="{plot_x - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="10.5" fill="#64748b">'
            f"{y_value:.4f}</text>"
        )

    start_epoch = points[0][x_key]
    end_epoch = points[-1][x_key]

    svg = f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">
      <rect x="0" y="0" width="{width}" height="{height}" rx="28" fill="#f8fafc" />
      <text x="{plot_x}" y="18" font-size="14.5" font-weight="700" fill="#0f172a">{escape(title)}</text>
      <line x1="{plot_x}" y1="{axis_y}" x2="{plot_x + plot_width}" y2="{axis_y}" stroke="#94a3b8" />
      <line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{axis_y}" stroke="#94a3b8" />
      {''.join(labels)}
      <polyline fill="none" stroke="{stroke}" stroke-width="2.6" points="{' '.join(polyline_points)}" />
      {''.join(markers)}
      <text x="{plot_x}" y="{axis_y + 20}" font-size="11" fill="#475569">epoch {start_epoch}</text>
      <text x="{plot_x + plot_width}" y="{axis_y + 20}" text-anchor="end" font-size="11" fill="#475569">
        epoch {end_epoch}
      </text>
    </svg>
    """
    return mark_safe(svg)
