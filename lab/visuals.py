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


def render_multi_line_chart(title, series, width=760, height=320):
    if not series:
        return ""

    plot_x = 60
    plot_y = 34
    plot_width = width - 96
    plot_height = height - 92
    axis_y = plot_y + plot_height
    palette = ["#0f766e", "#1d4ed8", "#c2410c", "#be123c", "#7c3aed", "#0891b2"]

    label_order = {}
    for item in series:
        for point in item.get("points", []):
            label = str(point.get("label", ""))
            if label not in label_order:
                label_order[label] = point.get("x")

    ordered_labels = [
        label
        for label, _ in sorted(
            label_order.items(),
            key=lambda item: (item[1] is None, item[1] if item[1] is not None else item[0]),
        )
    ]
    if not ordered_labels:
        return ""

    x_positions = {}
    for index, label in enumerate(ordered_labels):
        ratio = index / max(len(ordered_labels) - 1, 1)
        x_positions[label] = plot_x + plot_width * ratio

    values = [
        float(point["value"])
        for item in series
        for point in item.get("points", [])
        if point.get("value") is not None
    ]
    lower, upper = _dynamic_range(values)

    ticks = []
    for tick_idx in range(5):
        y_value = lower + (upper - lower) * (4 - tick_idx) / 4
        y = plot_y + plot_height * tick_idx / 4
        ticks.append(
            f'<line x1="{plot_x}" y1="{y:.2f}" x2="{plot_x + plot_width}" y2="{y:.2f}" '
            f'stroke="#cbd5e1" stroke-dasharray="4 6" />'
        )
        ticks.append(
            f'<text x="{plot_x - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="10.5" fill="#64748b">'
            f"{y_value:.4f}</text>"
        )

    x_labels = []
    for index, label in enumerate(ordered_labels):
        show_label = len(ordered_labels) <= 7 or index in {0, len(ordered_labels) // 2, len(ordered_labels) - 1}
        if not show_label:
            continue
        x_labels.append(
            f'<text x="{x_positions[label]:.2f}" y="{axis_y + 22:.2f}" text-anchor="middle" '
            f'font-size="10.5" fill="#475569">{escape(label)}</text>'
        )

    polylines = []
    markers = []
    legends = []
    for index, item in enumerate(series):
        color = palette[index % len(palette)]
        points = []
        for point in item.get("points", []):
            label = str(point.get("label", ""))
            value = point.get("value")
            if label not in x_positions or value is None:
                continue
            x = x_positions[label]
            y = axis_y - ((float(value) - lower) / (upper - lower)) * plot_height
            points.append((x, y))
            markers.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="{color}" />')

        if not points:
            continue

        polylines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.4" '
            f'points="{" ".join(f"{x:.2f},{y:.2f}" for x, y in points)}" />'
        )
        legend_y = 20 + index * 16
        legends.append(
            f'<g transform="translate({width - 170},{legend_y})">'
            f'<line x1="0" y1="7" x2="16" y2="7" stroke="{color}" stroke-width="2.4" />'
            f'<text x="22" y="10" font-size="10.5" fill="#334155">{escape(item.get("label", ""))}</text>'
            f"</g>"
        )

    svg = f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">
      <rect x="0" y="0" width="{width}" height="{height}" rx="28" fill="#f8fafc" />
      <text x="{plot_x}" y="20" font-size="14.5" font-weight="700" fill="#0f172a">{escape(title)}</text>
      <line x1="{plot_x}" y1="{axis_y}" x2="{plot_x + plot_width}" y2="{axis_y}" stroke="#94a3b8" />
      <line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{axis_y}" stroke="#94a3b8" />
      {''.join(ticks)}
      {''.join(polylines)}
      {''.join(markers)}
      {''.join(x_labels)}
      {''.join(legends)}
    </svg>
    """
    return mark_safe(svg)
